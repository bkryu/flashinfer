"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import defaultdict

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    print_perf_metrics,
    filter_backends_by_compute_capability,
)


def run_topk_test(args):
    """
    Run a top-k test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "top_k":
        return testTopK(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_topk_args(line, parser):
    """
    Parse command line arguments for top-k test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size (number of rows).",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Vocabulary size (number of columns in input tensor).",
    )
    parser.add_argument(
        "--top_k_value",
        "-k",
        type=int,
        required=True,
        help="Number of top elements to select.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type of the input tensor. Note: cute-dsl backend does not support float16.",
    )
    parser.add_argument(
        "--sorted",
        action="store_true",
        default=False,
        help="Return sorted results (descending order).",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["cuda", "cute-dsl"],
        choices=["cuda", "cute-dsl"],
        help="Backends to test. Default: cuda cute-dsl",
    )

    args = parser.parse_args(line)

    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def testTopK(args):
    """
    Test top_k API.

    This test:
    1. Generates random input tensors
    2. Runs top_k with both cuda and cute-dsl backends
    3. Runs reference check (comparing top-k values against torch.topk)
    4. Measures performance metrics (memory bandwidth)

    Note: Top-K is memory-bandwidth bound, so TB/sec is the primary metric.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTopK")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]  # Make a copy to avoid modifying the original
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    k = args.top_k_value
    sorted_output = args.sorted
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    # Validate k
    if k > vocab_size:
        raise ValueError(f"top_k_value ({k}) must be <= vocab_size ({vocab_size})")

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are bfloat16, float16, float32."
        )

    # cute-dsl backend does not support float16
    if input_dtype == torch.float16 and "cute-dsl" in backends:
        print(
            "[WARNING] cute-dsl backend does not support float16. Removing cute-dsl from backends."
        )
        backends.remove("cute-dsl")
        if len(backends) == 0:
            print("[ERROR] No backends to test. Exiting.")
            return res

    ## Done parsing input arguments

    ## Prepare input tensors
    input_shape = (batch_size, vocab_size)
    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {k = }")
        print(f"[VVERBOSE] {sorted_output = }")

    def run_backend(backend, input_tensor):
        if backend == "cuda":
            return flashinfer.top_k(input_tensor, k, sorted=sorted_output, backend="cuda")
        elif backend == "cute-dsl":
            return flashinfer.top_k(
                input_tensor, k, sorted=sorted_output, backend="cute-dsl"
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Reference: PyTorch top_k
    has_reference_output = False
    reference_values = None
    if run_refcheck:
        reference_values, reference_indices = torch.topk(
            input_tensor, k, dim=-1, largest=True, sorted=sorted_output
        )
        has_reference_output = True

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            values, indices = run_backend(cur_backend, input_tensor)
            outputs[cur_backend] = (values.detach().clone(), indices.detach().clone())
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, input_tensor),
        )

    tested_backends = list(outputs.keys())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                values, indices = outputs[tested_backends[i]]
                if args.verbose >= 2:
                    print(
                        f"[VVERBOSE] Backend {tested_backends[i]}: "
                        f"values.shape = {values.shape}, values.dtype = {values.dtype}, "
                        f"indices.shape = {indices.shape}, indices.dtype = {indices.dtype}"
                    )

                # For top-k, we compare values (indices might differ due to tie-breaking)
                # Sort both to compare the actual top-k values found
                ref_sorted = torch.sort(reference_values, dim=-1, descending=True)[0]
                out_sorted = torch.sort(values, dim=-1, descending=True)[0]

                # Compare sorted values
                max_diff = (ref_sorted.float() - out_sorted.float()).abs().max().item()
                if args.verbose >= 2:
                    print(
                        f"[VVERBOSE] Backend {tested_backends[i]}: max value diff = {max_diff:.6f}"
                    )

                # Allow some tolerance for floating point comparisons
                rtol = 1e-3 if input_dtype == torch.float32 else 1e-2
                atol = 1e-3 if input_dtype == torch.float32 else 1e-2
                values_match = torch.allclose(ref_sorted, out_sorted, rtol=rtol, atol=atol)

                if not values_match:
                    mismatch_msg = (
                        f"[top_k] Backend {tested_backends[i]}: "
                        f"top-k values mismatch with reference (max diff: {max_diff:.6f})"
                    )
                    if args.allow_output_mismatch:
                        print(f"[WARNING] {mismatch_msg}")
                    else:
                        raise AssertionError(mismatch_msg)

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation for top-k
            # Read: input tensor
            # Write: values tensor (k elements per row) + indices tensor (k elements per row)
            num_input_elements = batch_size * vocab_size
            num_output_elements = batch_size * k
            # Indices are int32 (4 bytes) for cuda backend, int64 for cute-dsl
            indices_dtype_size = 8  # long (int64)
            problem_bytes = (
                num_input_elements * input_dtype.itemsize  # input read
                + num_output_elements * input_dtype.itemsize  # values write
                + num_output_elements * indices_dtype_size  # indices write
            )
            # Top-k is memory-bound; rough FLOPS estimate (comparisons + swaps)
            # Radix-based: ~O(N * log2(max_value)) comparisons per element
            # For simplicity, estimate as N * bits_per_element
            problem_flops = num_input_elements * 16  # rough estimate

            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["batch_size"] = batch_size
                cur_res["vocab_size"] = vocab_size
                cur_res["top_k_value"] = k
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["sorted"] = sorted_output
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res

