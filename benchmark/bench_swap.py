"""
Benchmark prefill latency for four KV-cache scenarios:

  cold      -- all tokens computed from scratch (no cache)
  gpu-hit   -- prefix KV already in GPU pool; compute only new tokens
  host-hit  -- prefix KV in CPU RAM via LMCache (local_cpu backend)
  gds-hit   -- prefix KV on disk via LMCache (GDS backend)

Usage:
  python bench_swap.py \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct --load-format dummy \
    --prefix-len 512 --new-len 128 \
    [--gds-path /tmp/lmcache_bench --chunk-size 256]

Requirements:
    - LMCache: v0.4.5
"""

import argparse
import dataclasses
import logging
import multiprocessing
import itertools
import os
import shutil
import time
import tempfile
import numpy as np
import torch
from contextlib import contextmanager
from typing import List, Tuple

from sglang.bench_one_batch import load_model

from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    kill_process_tree,
    maybe_reindex_device_id,
    set_gpu_proc_affinity,
)

try:
    from lmcache.integration.sglang.sglang_adapter import (
        LMCacheLayerwiseConnector,
        LoadMetadata,
        StoreMetadata,
    )
    from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
        LayerTransferCounter, )

    _HAS_LMCACHE = True
except ImportError:
    _HAS_LMCACHE = False


@dataclasses.dataclass
class BenchArgs:
    prefix_len: Tuple[int, ...] = (512,)
    new_len: Tuple[int, ...] = (128,)
    num_tests: int = 5
    num_warmups: int = 2
    gds_path: str = ""
    chunk_size: int = 256

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--prefix-len", type=int, nargs="+", default=[512])
        parser.add_argument("--new-len", type=int, nargs="+", default=[128])
        parser.add_argument("--num-tests", type=int, default=5)
        parser.add_argument("--num-warmups", type=int, default=2)
        parser.add_argument(
            "--gds-path",
            type=str,
            default="",
            help="LMCache GDS storage directory. Enables gds-hit.",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=256,
            help="LMCache chunk size; prefix-len must be a multiple.",
        )

    @classmethod
    def from_cli_args(cls, args):
        attrs = [(f.name, type(f.default)) for f in dataclasses.fields(cls)]
        result = {}
        for attr, attr_type in attrs:
            value = getattr(args, attr)
            if value is None or isinstance(attr_type, type(None)):
                result[attr] = value
            else:
                result[attr] = attr_type(value)
        return cls(**result)


def _lmcache_reset():
    try:
        from lmcache.v1.cache_engine import LMCacheEngineBuilder

        LMCacheEngineBuilder._instances.clear()
    except Exception:
        pass


@contextmanager
def _lmc_connector(
    model_config,
    kv_pool,
    chunk_size,
    use_gds,
    gds_path="",
    gds_buffer_size=0,
):
    config_tmp = tempfile.mkdtemp(
        prefix="lmcache_gds_" if use_gds else "lmcache_cpu_")

    # ceil(bytes / MiB)
    gds_buffer_size_mb = (gds_buffer_size + (1 << 20) - 1) >> 20
    gds_buffer_size_mb = max(256, gds_buffer_size_mb)

    connector = None
    try:
        _lmcache_reset()
        config_path = os.path.join(config_tmp, "_bench.yaml")
        lines = [f"chunk_size: {chunk_size}\n", "use_layerwise: true\n"]
        lines += ([
            "local_cpu: false\n",
            f'gds_path: "{gds_path}"\n',
            f"gds_buffer_size: {gds_buffer_size_mb}\n",
            "extra_config: {'use_direct_io': true}",
        ] if use_gds else ["local_cpu: true\n", "max_local_cpu_size: 10\n"])
        with open(config_path, "w") as f:
            f.writelines(lines)
        os.environ["LMCACHE_CONFIG_FILE"] = config_path
        os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"

        connector = LMCacheLayerwiseConnector(
            sgl_config=model_config,
            tp_size=1,
            rank=0,
            k_pool=kv_pool.k_buffer,
            v_pool=kv_pool.v_buffer,
        )

        torch.cuda.current_stream().synchronize()
        yield connector
    finally:
        if connector is not None:
            connector.close()
        _lmcache_reset()
        shutil.rmtree(config_tmp, ignore_errors=True)


def _synthetic_ids(batch_size, prefix_len, new_len):
    rng = np.random.default_rng(42)
    ids = rng.integers(0,
                       10000, (batch_size, prefix_len + new_len),
                       dtype=np.int32)
    prefix_ids = [list(row[:prefix_len]) for row in ids]
    new_ids = [list(row[prefix_len:]) for row in ids]
    full_ids = [list(row) for row in ids]
    return prefix_ids, new_ids, full_ids


def _make_reqs(ids_list):
    sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
    reqs = []
    for i, ids in enumerate(ids_list):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(ids),
            sampling_params=sampling_params,
        )
        req.fill_ids = list(ids)
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)
    return reqs


def _make_hit_reqs(ids_list, prefix_ids_list, kv_allocator):
    kv_pool = kv_allocator.get_kvcache()
    if not hasattr(kv_pool, "k_buffer"):
        raise RuntimeError(
            "bench_cache_hit requires MHATokenToKVPool (k_buffer).")

    hit_reqs = _make_reqs(ids_list)
    assert len(hit_reqs) == len(prefix_ids_list)

    store_md_list: List[StoreMetadata] = []
    load_md_list: List[LoadMetadata] = []
    for (req, prefix_ids) in zip(hit_reqs, prefix_ids_list):
        kv_indices = kv_allocator.alloc(need_size=len(prefix_ids))
        req.prefix_indices = kv_indices.detach().clone()
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))

        store_md_list.append(
            StoreMetadata(last_node=None,
                          token_ids=prefix_ids,
                          kv_indices=req.prefix_indices,
                          offset=0))

        load_md_list.append(
            LoadMetadata(token_ids=prefix_ids,
                         slot_mapping=req.prefix_indices,
                         offset=0))

    return hit_reqs, store_md_list, load_md_list


def _measure_extend(model_runner, reqs, expected_output_ids=None):
    model_runner.synchronize()
    start = time.perf_counter()
    next_token_ids, _, batch = model_runner.extend(reqs)
    model_runner.synchronize()
    lat = time.perf_counter() - start
    model_runner.cleanup(batch)
    if expected_output_ids is not None:
        if not torch.equal(next_token_ids, expected_output_ids):
            logging.warning("[correctness] output token mismatch detected")
    return lat


def _bench_lmcache_hit(
    model_runner,
    full_ids,
    prefix_ids,
    chunk_size,
    gds_path,
    gds_buffer_size,
    num_tests,
    num_warmups,
    expected_output_ids,
):
    """
    return List[latency_ms]
    """
    raw = model_runner.torch_runner
    model_config = raw.model_config
    kv_allocator: TokenToKVPoolAllocator = raw.token_to_kv_pool_allocator
    kv_pool = kv_allocator.get_kvcache()
    if not hasattr(kv_pool, "k_buffer"):
        raise RuntimeError(
            "bench_cache_hit requires MHATokenToKVPool (k_buffer).")

    bench_set = [(False, "host")]
    if gds_path:
        bench_set.append((True, "storage"))

    results = {}
    for use_gds, key in bench_set:
        lmc_kwargs = {
            "model_config": model_config,
            "kv_pool": kv_pool,
            "chunk_size": chunk_size,
            "use_gds": use_gds
        }
        if use_gds:
            lmc_kwargs["gds_path"] = gds_path

            def _next_power_of_two(n: int) -> int:
                if n <= 0:
                    return 1
                return 1 << n.bit_length()
            lmc_kwargs["gds_buffer_size"] = _next_power_of_two(gds_buffer_size)

        with _lmc_connector(**lmc_kwargs) as connector:
            store_stream = torch.cuda.Stream()
            load_stream = torch.cuda.Stream()
            counter = LayerTransferCounter(
                num_layers=kv_pool.layer_num,
                load_stream=load_stream,
                lmc_connector=connector,
            )
            kv_pool.register_layer_transfer_counter(counter)

            host_hit_lats = []
            for _ in range(num_warmups + num_tests):
                model_runner.clear()
                hit_reqs, store_md_list, load_md_list = _make_hit_reqs(
                    full_ids, prefix_ids, kv_allocator)

                with torch.cuda.stream(store_stream):
                    for store_md in store_md_list:
                        connector.store_kv(store_md)

                store_stream.synchronize()

                for load_md in load_md_list:
                    connector.start_load_kv(load_md)

                host_hit_lats.append(
                    _measure_extend(model_runner, hit_reqs,
                                    expected_output_ids))

            kv_pool.register_layer_transfer_counter(None)

            lat_s = float(np.mean(host_hit_lats[num_warmups:]))
            results[key] = lat_s

    return results


def bench_one_config(
    model_runner,
    prefix_len,
    new_len,
    num_tests,
    num_warmups,
    gds_path="",
    chunk_size=256,
) -> dict:
    batch_size = 1
    raw = model_runner.torch_runner
    kv_allocator: TokenToKVPoolAllocator = raw.token_to_kv_pool_allocator
    kv_pool = kv_allocator.get_kvcache()
    if not hasattr(kv_pool, "k_buffer"):
        raise RuntimeError(
            "bench_cache_hit requires MHATokenToKVPool (k_buffer).")

    num_layers = kv_pool.layer_num
    kv_size_b = (batch_size * prefix_len * num_layers * 2 * kv_pool.head_num *
                 kv_pool.head_dim * kv_pool.store_dtype.itemsize)

    prefix_ids, new_ids, full_ids = _synthetic_ids(batch_size, prefix_len,
                                                   new_len)

    n_total = num_warmups + num_tests
    results = {
        "prefix_len": prefix_len,
        "new_len": new_len,
        "kv_size_mb": kv_size_b / (1 << 20),
    }

    # Get output_ids for correctness evaluation in the tests below
    def _get_output_ids(model_runner, input_ids):
        model_runner.clear()
        reqs = _make_reqs(input_ids)
        output_ids, _, batch = model_runner.extend(reqs)
        model_runner.synchronize()
        model_runner.cleanup(batch)
        return output_ids

    expected_output_ids = _get_output_ids(model_runner, full_ids)

    # Recompuation latency measurement
    recompute_lats = []
    for _ in range(n_total):
        model_runner.clear()
        reqs = _make_reqs(full_ids)
        recompute_lats.append(
            _measure_extend(model_runner, reqs, expected_output_ids))

    lat_s = float(np.mean(recompute_lats[num_warmups:]))
    results["recompute_lat_ms"] = lat_s * 1e3

    # GPU hit lateny measurement
    gpu_hit_lats = []
    for _ in range(n_total):
        model_runner.clear()
        hit_reqs, _, _ = _make_hit_reqs(full_ids, prefix_ids, kv_allocator)
        gpu_hit_lats.append(
            _measure_extend(model_runner, hit_reqs, expected_output_ids))

    lat_s = float(np.mean(gpu_hit_lats[num_warmups:]))
    results["gpu_hit_lat_ms"] = lat_s * 1e3

    # LMCache(host and storage) hit latency measurement
    if _HAS_LMCACHE:
        lmc_results = _bench_lmcache_hit(model_runner, full_ids, prefix_ids,
                                         chunk_size, gds_path, kv_size_b,
                                         num_tests, num_warmups,
                                         expected_output_ids)
        for key, lat_s in lmc_results.items():
            results[f"{key}_hit_latency"] = lat_s * 1e3
            results[f"{key}_bw_gb_s"] = (kv_size_b / (1 << 30)) / lat_s

    return results


def _print_results_table(results: list[dict]):
    """Print benchmark results in a clean aligned table."""
    if not results:
        return

    keys = list(results[0].keys())

    header = ""
    wspace = 3
    for key in keys:
        width = len(key) + wspace
        header += f"{key:>{width}}"
    header = header[wspace:]  # Strip leading whitespace

    print(header)
    print("-" * len(header))

    for result in results:
        row = ""
        for key in keys:
            value = result.get(key, float("nan"))
            width = len(key) + wspace
            if isinstance(value, float):
                row += f"{value:>{width}.2f}"
            else:
                row += f"{value:>{width}}"
        row = row[wspace:]  # Align columns with the header
        print(row)


def swap_latency_test(
    server_args,
    port_args,
    bench_args,
    gpu_id,
    tp_rank,
):
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)

    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size,
            server_args.tp_size,
            server_args.nnodes,
            tp_rank,
        )

    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    model_runner, tokenizer = load_model(server_args, port_args, gpu_id,
                                         tp_rank)

    rank_print("\nBenchmark ...\n")
    result_list = []
    for prefix_len, new_len in itertools.product(
            bench_args.prefix_len, bench_args.new_len):
        try:
            ret = bench_one_config(
                model_runner,
                prefix_len,
                new_len,
                bench_args.num_tests,
                bench_args.num_warmups,
                gds_path=bench_args.gds_path,
                chunk_size=bench_args.chunk_size,
            )
            result_list.append(ret)
        except Exception as e:
            import traceback

            rank_print("[SKIP] prefix={} new={}: {}\n{}".format(
                prefix_len, new_len, e, traceback.format_exc()))

    if result_list and tp_rank == 0:
        _print_results_table(result_list)

    if server_args.tp_size > 1:
        destroy_distributed_environment()


def main(server_args, bench_args):
    # Disable capturing CUDA graph
    server_args.disable_cuda_graph = True

    _set_envs_and_config(server_args)

    if server_args.model_path:
        work_func = swap_latency_test
    else:
        raise ValueError("Provide --model-path for running the tests or "
                         "provide --result-filename for plotting the results")

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            with maybe_reindex_device_id(tp_rank) as gpu_id:
                proc = multiprocessing.Process(
                    target=work_func,
                    args=(
                        server_args,
                        port_args,
                        bench_args,
                        gpu_id,
                        tp_rank,
                    ),
                )
                proc.start()
                workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
