# FlashInfer API Logging

FlashInfer provides comprehensive API logging to help debug issues, analyze performance, and reproduce crashes. This document describes all available logging levels and their features.

## Quick Start

Enable logging using two environment variables:

```bash
# Set logging level (0-10)
export FLASHINFER_LOGLEVEL_DBG=3

# Set log destination (default is stdout)
export FLASHINFER_LOGDEST_DBG=stdout  # or stderr, or a file path like "flashinfer.log"

# Run your code
python train.py
```

## Logging Levels

| Level | Name | Features | Use Case |
|-------|------|----------|----------|
| **0** | Disabled (Default) | No logging (zero overhad) | Production |
| **1** | Function Names | Function names only | Basic tracing |
| **3** | Inputs/Outputs | Function names + arguments + outputs with metadata | Standard debugging |
| **5** | Statistics | Level 3 + tensor statistics (min, max, mean, NaN/Inf counts) | Numerical analysis |
| **8** | Library Logs | Level 5 + cuDNN/cuBLAS/cuBLASLt API logging | Dependency  debugging |
| **10** | Tensor Dumps | Level 8 + dump all tensors to disk | Crash reproduction |

## Detailed Level Descriptions

### Level 8: Library API Logging

Level 5 + automatic cuDNN/cuBLAS/cuBLASLt API logging.

**What it does:**
- Automatically enables cuBLAS, cuBLASLt, cuDNN backend, and cuDNN frontend logging
- Logs written to separate files: `flashinfer_cublas_log_<pid>.txt`, `flashinfer_cublaslt_log_<pid>.txt`, `flashinfer_cudnn_backend_log_<pid>.txt`, and `flashinfer_cudnn_frontend_log_<pid>.txt`.
- Respects user configuration (won't override if you've already set library logging)


### Level 10: Tensor Dumps

**Level 8 + dump all input and output tensors to disk** (uses PyTorch's native serialization).

Three optional environment variables are used in level 10:
```bash
# Optional: configure dump settings
export FLASHINFER_DUMP_DIR=flashinfer_dumps       # default: flashinfer_dumps
export FLASHINFER_DUMP_MAX_SIZE_GB=20             # default: 20
export FLASHINFER_DUMP_MAX_COUNT=1000             # default: 1000
```

**What gets saved:**

Each function call creates a directory:
```
flashinfer_dumps/
└── 20251120_101530_789_mm_fp4_call0001/
    ├── inputs.pt              # All input tensors (stride preserved)
    ├── outputs.pt             # All output tensors (if function completes)
    └── metadata.json          # Function info, shapes, dtypes, etc.
```

**Example metadata.json:**
```json
{
  "function_name": "test_matmul_decorated",
  "module": "__main__",
  "call_sequence": 4,
  "timestamp": "20251119_160153_217",
  "process_id": 452930,
  "input_metadata": {},
  "output_metadata": {},
  "tensor_info": {
    "input_tensor_keys": [
      "arg_0",
      "arg_1"
    ],
    "output_tensor_keys": [
      "result"
    ],
    "total_size_bytes": 3145728,
    "total_size_mb": 3.0
  },
  "function_signature": "(A, B)",
  "versions": {
    "torch": "2.9.0+cu130",
    "python": "3.12.3 (main, Aug 14 2025, 17:47:21) [GCC 13.3.0]",
    "flashinfer": "0.5.2"
  }
}
```

**Replay dumps:**
```python
from flashinfer.api_logging import replay_from_dump

# Load dumped data (tensors moved to cuda:0 by default)
data = replay_from_dump("flashinfer_dumps/20251120_101530_789_mm_fp4_call0001/")

# Or load to specific device
data = replay_from_dump("flashinfer_dumps/.../")

# data contains:
# - data['args']: positional arguments (list) - tensors on specified device
# - data['kwargs']: keyword arguments (dict) - tensors on specified device
# - data['metadata']: full metadata (dict)

# Reproduce the call
from flashinfer.gemm import bmm_fp8
result = bmm_fp8(*data['args'], **data['kwargs'])

# Note: Tensor stride and contiguity are preserved during save/load
```

## Environment Variables

### Main Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FLASHINFER_LOGLEVEL_DBG` | int | 0 | Logging level (0, 1, 3, 5, 8, or 10) |
| `FLASHINFER_LOGDEST_DBG` | str | `stdout` | Log destination: `stdout`, `stderr`, or file path |

### Level 10 Tensor Dumping

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FLASHINFER_DUMP_DIR` | str | `flashinfer_dumps` | Directory for tensor dumps |
| `FLASHINFER_DUMP_MAX_SIZE_GB` | float | 20 | Maximum total size of dumps (GB) |
| `FLASHINFER_DUMP_MAX_COUNT` | int | 1000 | Maximum number of function call dumps |

### Process ID Substitution

Use `%i` in file paths for automatic process ID substitution (useful for multi-GPU training):

```bash
export FLASHINFER_LOGDEST_DBG="flashinfer_log_%i.txt"  # → flashinfer_log_12345.txt
```

This works for:
- `FLASHINFER_LOGDEST_DBG`
- All cuDNN/cuBLAS/cuBLASLt log destinations (at Level 8)

## Advanced Features

### Crash Safety

**All logging levels are crash-safe:**
- Level 1: Function name logged BEFORE execution
- Level 3+: Inputs logged BEFORE execution
- Level 10: Tensors dumped BEFORE execution

If a CUDA kernel crashes, you'll still have:
- Function name (Level 1+)
- Input arguments and metadata (Level 3+)
- Input tensors for reproduction (Level 10)

**Example crash scenario:**
```bash
export FLASHINFER_LOGLEVEL_DBG=10
python train.py  # Crashes with "CUDA error: illegal memory access"

# Inputs are still saved!
ls flashinfer_dumps/20251120_101530_789_mm_fp4_call0042/
# → inputs.pt  ✓
# → metadata.json (execution_status="inputs_saved")  ✓
# → outputs.pt  ✗ (not created - function crashed)
```

### CUDA Graph Compatibility

Level 5 statistics are **automatically skipped during CUDA graph capture** to avoid synchronization issues.

```python
# This works correctly - no synchronization errors
with torch.cuda.graph(cuda_graph):
    result = mm_fp4(a, b, scales)  # Level 5 logging active
    # Statistics automatically skipped during capture
```

Output shows: `[statistics skipped: CUDA graph capture in progress]`

### Enum Support

Enum arguments are logged with both name and value:

```python
from enum import Enum

class Mode(Enum):
    FAST = 1
    ACCURATE = 2

@flashinfer_api_log
def process(data, mode: Mode):
    ...

# Log output shows: mode= Mode.FAST (value=1)
```

### Default Parameter Detection

Parameters using default values are explicitly marked:

```python
@flashinfer_api_log
def compute(x, y, dtype=torch.float16):
    ...

compute(a, b)  # dtype not provided

# Log shows: dtype= torch.float16 [default]
```

### System Information Logging

At Level 1+, system information is logged once at startup:
- FlashInfer version
- CUDA toolkit version
- cuDNN version
- GPU name(s) and compute capability
- PyTorch version
- Library logging status (Level 8+)

## Examples

### Example 1: Basic Debugging

```bash
export FLASHINFER_LOGLEVEL_DBG=3
export FLASHINFER_LOGDEST_DBG=stdout

python awesome_script_that_uses_FlashInfer.py
```

### Example 2: Find NaN Issues

```bash
export FLASHINFER_LOGLEVEL_DBG=5
export FLASHINFER_LOGDEST_DBG=debug.log

python awesome_script_that_uses_FlashInfer.py
grep "nan_count" debug.log  # Find where NaNs appear
```

### Example 3: Reproduce a Crash

```bash
# Run with tensor dumping
export FLASHINFER_LOGLEVEL_DBG=10
python awesome_script_that_uses_FlashInfer.py  # Crashes

# Find the dump
ls flashinfer_dumps/
# → 20251120_101530_789_mm_fp4_call0042/

# Reproduce locally (tensors moved to cuda:0)
python -c "
from flashinfer.api_logging import replay_from_dump
from flashinfer import mm_fp4

data = replay_from_dump('flashinfer_dumps/20251120_101530_789_mm_fp4_call0042/')
# Tensors loaded to cuda:0 (default, with stride/contiguity preserved)
result = mm_fp4(*data['args'], **data['kwargs'])  # Should crash the same way
"
```

### Example 4: Multi-GPU Training

```bash
# Use %i for process ID substitution
export FLASHINFER_LOGLEVEL_DBG=3
export FLASHINFER_LOGDEST_DBG="logs/flashinfer_api_%i.log"

torchrun --nproc_per_node=8 awesome_script_that_uses_FlashInfer.py

# Creates separate logs:
# logs/flashinfer_api_12345.log (rank 0)
# logs/flashinfer_api_12346.log (rank 1)
# ...
```

### Example 5: Dependency Debugging

```bash
export FLASHINFER_LOGLEVEL_DBG=8

python awesome_script_that_uses_FlashInfer.py

# Check which cuBLAS/cuDNN kernels are used
cat flashinfer_cublas_log_*.txt
cat flashinfer_cudnn_backend_log_*.txt
```

## Frequently Asked Questions

### Q: Does Level 0 really have zero overhead?

**A: Yes.** At Level 0, the decorator returns the original function unchanged. No wrapper, no checks, no overhead.

### Q: Can I change the logging level at runtime?

**A: No.** The logging level is read once when the `api_logging` module is imported. You must restart your process to change levels.

### Q: Why are levels 2, 4, 6, 7, 9 skipped?

**A: Reserved for future features.** The remapping (0→0, 1→1, 2→3, 3→5) leaves room for intermediate levels without using fractional numbers.

### Q: Do I need to install anything extra?

**A: No.** All logging levels, including Level 10 tensor dumping, work with the base FlashInfer and PyTorch installation. No additional dependencies are required.

### Q: How do I know if library logging (Level 8) is working?

**A: Check the log files:**

```bash
export FLASHINFER_LOGLEVEL_DBG=8
python train.py

# Check that files were created
ls flashinfer_cublas_log_*.txt
ls flashinfer_cublaslt_log_*.txt
ls flashinfer_cudnn_backend_log_*.txt
ls flashinfer_cudnn_frontend_log_*.txt
```

### Q: Can I override the library logging settings?

**A: Yes.** If you set the "switch" environment variable for a library before importing FlashInfer, your settings will be respected:

```bash
# Disable cuBLAS logging even at Level 8
export CUBLAS_LOGINFO_DBG=0

export FLASHINFER_LOGLEVEL_DBG=8
python train.py  # cuBLAS logging disabled, others enabled
```

### Q: What if a dump exceeds the size limit?

**A: The dump is skipped.** You'll see a warning in the logs. Increase the limit:

```bash
export FLASHINFER_DUMP_MAX_SIZE_GB=50  # Increase to 50GB
```

### Q: What happens with non-contiguous tensors?

**A: Stride and contiguity information is fully preserved.** PyTorch's `torch.save/torch.load` preserves the exact memory layout, including non-contiguous tensors.

**Behavior:**
- Non-contiguous tensors (e.g., from `.transpose()`, `.T`, slicing) are saved with their exact stride
- Your original tensors are NOT modified
- When replayed, tensors have the **identical memory layout** as the original
- This ensures perfect reproduction, even for APIs sensitive to tensor strides

**Example:**
```python
# Your code has a non-contiguous tensor
B = A.transpose(0, 1)  # Non-contiguous view
result = bmm_fp8(x, B)  # Level 10 logging

# When replaying:
data = replay_from_dump("...")
B_loaded = data['kwargs']['B']
print(B_loaded.is_contiguous())  # False (same as original!)
print(B_loaded.stride())  # Identical stride to original B

# Perfect reproduction
result = bmm_fp8(*data['args'], **data['kwargs'])
```

### Q: How do I clean up old dumps?

**A: Manually delete the dump directory:**

```bash
rm -rf flashinfer_dumps/
```

Or use your own cleanup script based on timestamps.

### Q: What device are tensors loaded to when replaying?

**A: By default, all tensors are loaded to `cuda:0` (or you can specify a different device).**

```python
# Load to cuda:0 (default)
data = replay_from_dump("dump_dir/")

# Load to specific GPU
data = replay_from_dump("dump_dir/", device="cuda:1")

# Load to CPU
data = replay_from_dump("dump_dir/", device="cpu")
```

**Note:** Even if your original tensors were on different devices (cuda:1, cuda:2, etc.), 
they are all loaded to the specified device. This simplifies replay in most debugging scenarios.

## See Also

- [API Documentation](https://docs.flashinfer.ai/)
- [Benchmark Scripts](benchmarks/README.md)
- [Contributing Guide](CONTRIBUTING.md)

## Summary

FlashInfer's API logging provides flexible, crash-safe debugging at multiple levels:

- **Level 0**: Production (zero overhead)
- **Level 1**: Basic tracing (<0.1% overhead)
- **Level 3**: Standard debugging (~1% overhead)
- **Level 5**: Numerical analysis (~2-5% overhead)
- **Level 8**: Performance debugging (~5-10% overhead)
- **Level 10**: Crash reproduction (~10-50% overhead)

Choose the level that matches your debugging needs and accept the corresponding overhead.

