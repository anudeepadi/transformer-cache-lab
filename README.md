# Transformer Cache Lab

A NumPy experiment comparing memory layouts in multi-head attention. The implementation exposes contiguous Q/K/V storage, row/column layout and alignment options, with a small test suite and saved benchmark artifacts.

**Status:** educational experiment from December 2024. This is an attention-layout study; it does not implement a production transformer serving engine or a general KV-cache library.

## Run a small example

```bash
git clone https://github.com/anudeepadi/transformer-cache-lab.git
cd transformer-cache-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pytest
python -m pytest tests/test_attention.py
```

```python
import numpy as np
from src.model.attention import CacheAwareAttention, MemoryLayout

np.random.seed(42)
model = CacheAwareAttention(dim=8, num_heads=2,
    memory_layout=MemoryLayout(contiguous_qkv=True, row_major=True))
output = model.forward(np.ones((2, 4, 8)))
print(output.shape)  # (2, 2, 4, 4)
```

Validation on 22 September 2026: the four existing attention tests passed, and the example above produced `(2, 2, 4, 4)`. This validates the small example, not performance claims.

## Saved measurements

![Historical attention execution times](benchmark_results/execution_times.png)

The [saved JSON](benchmark_results/benchmark_results.json) and [benchmark script](benchmarks/benchmark_layouts.py) cover small combinations of model dimension, batch size, sequence length and layout. The JSON does not record hardware, software versions or run date; several memory measurements are zero. These artifacts cannot support a general percentage memory reduction or inference-speed claim.

To reproduce the benchmark, install `requirements.txt` and run `python -m benchmarks.benchmark_layouts` from the repository root. Record CPU, OS, NumPy/BLAS versions and raw timings alongside any comparison. Profiling uses Windows counters or Linux `perf`; the current non-Windows path also selects Linux `perf` on macOS and needs platform-specific handling there. Counter-derived cache statistics require validation before interpretation.

## Source map

- [Attention and layout options](src/model/attention.py)
- [Shape/layout/numerical-finiteness tests](tests/test_attention.py)
- [Profiling helpers](src/profiling/cache_monitor.py)

The tests check output shape and finite values. They do not establish equivalence to a reference attention implementation or benchmark statistical significance.

## Credits

The original project acknowledges transformer research, the PyTorch community, ML optimization research and contributing developers. The attention implementation demonstrated above uses NumPy.
