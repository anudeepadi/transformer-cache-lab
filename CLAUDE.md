# CLAUDE.md - AI Assistant Guide for Cache_OPT_Transformers

## Project Overview

**Cache_OPT_Transformers** is a specialized framework for optimizing cache performance in Transformer models, focusing on efficient memory usage and faster inference times through advanced caching strategies for attention mechanisms.

**Key Focus**: CPU cache optimization for small-scale transformer models with emphasis on memory layout patterns, cache-aware attention computation, and performance profiling.

## Repository Structure

```
cache_opt/
├── src/
│   ├── model/
│   │   └── attention.py          # Cache-aware attention implementation
│   └── profiling/
│       └── cache_monitor.py      # Performance monitoring and profiling tools
├── tests/
│   └── test_attention.py         # Unit tests for attention module
├── benchmarks/
│   └── benchmark_layouts.py      # Memory layout benchmarking scripts
├── benchmark_results/
│   ├── benchmark_results.json    # Performance metrics data
│   ├── cache_hit_rates.png       # Cache performance visualizations
│   ├── execution_times.png       # Timing benchmarks
│   └── memory_usage.png          # Memory usage analysis
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation configuration
└── README.md                     # User-facing documentation
```

## Core Components

### 1. Cache-Aware Attention (`src/model/attention.py`)

**Primary Classes**:
- `CacheAwareAttention`: Main attention implementation with cache optimization
- `MemoryLayout`: Configuration dataclass for memory layout patterns

**Key Features**:
- Multi-head attention with configurable cache strategies
- Memory layout optimization (row-major vs column-major)
- Cache-aligned memory allocation
- Block-based tiling for cache efficiency
- Dynamic sequence length handling

**Configuration Options**:
```python
MemoryLayout(
    contiguous_qkv: bool = True,    # Q,K,V matrix storage pattern
    row_major: bool = True,          # Memory storage order
    cache_aligned: bool = False,     # Cache line alignment
    block_size: int = 64            # Tiling block size
)
```

### 2. Performance Monitoring (`src/profiling/cache_monitor.py`)

**Primary Classes**:
- `CacheMonitor`: Cross-platform cache performance monitoring
- `CacheStats`: Container for cache performance metrics
- `WindowsPerfCounter`: Windows-specific performance counters
- `LinuxPerfCounter`: Linux perf integration

**Metrics Tracked**:
- L1/L2 cache hit rates
- Memory access patterns
- CPU cycles and instructions
- Instructions per cycle (IPC)

**Decorator Available**:
```python
@profile_memory_access
def your_function():
    # Automatically profiles cache performance
    pass
```

### 3. Benchmarking System (`benchmarks/benchmark_layouts.py`)

**Purpose**: Systematic testing of different memory layouts and configurations

**Configuration Parameters**:
- `model_dims`: Model dimension sizes to test
- `batch_sizes`: Batch size variations
- `seq_lengths`: Sequence length variations
- `num_heads`: Number of attention heads

**Output**:
- JSON results with detailed metrics
- Visualization plots (execution time, memory usage, cache hit rates)

## Development Workflows

### Setup and Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_attention.py

# Run with verbose output
pytest tests/ -v

# Run specific test function
pytest tests/test_attention.py::test_attention_forward
```

### Running Benchmarks

```bash
# Run full benchmark suite
python benchmarks/benchmark_layouts.py

# Results saved to:
# - benchmark_results/benchmark_results.json
# - benchmark_results/*.png (plots)
```

### Performance Profiling

```python
from src.profiling.cache_monitor import CacheMonitor, profile_memory_access

# Method 1: Using decorator
@profile_memory_access
def my_attention_function():
    # Your code here
    pass

# Method 2: Manual monitoring
monitor = CacheMonitor()
monitor.start_monitoring()
# ... your code ...
stats = monitor.stop_monitoring()
print(f"L1 Hit Rate: {stats.l1_hit_rate:.2%}")
```

## Key Conventions for AI Assistants

### Code Style

1. **Follow PEP 8**: Standard Python style guidelines
2. **Type Hints**: Use type annotations for function signatures
3. **Dataclasses**: Prefer dataclasses for configuration objects
4. **Docstrings**: Use clear docstrings for classes and complex functions

### File Organization

- **Models**: Place in `src/model/`
- **Profiling/Monitoring**: Place in `src/profiling/`
- **Tests**: Mirror source structure in `tests/`
- **Benchmarks**: Standalone scripts in `benchmarks/`
- **Results**: Output data to `benchmark_results/`

### Testing Requirements

1. **All new features** must have corresponding unit tests
2. **Test naming**: `test_<feature_name>()` convention
3. **Test assertions**: Use descriptive error messages
4. **Edge cases**: Test boundary conditions (dimension mismatches, empty inputs)
5. **Numerical stability**: Verify no NaN/Inf values in outputs

Example test structure:
```python
def test_feature_name():
    # Setup
    model = CacheAwareAttention(dim=8, num_heads=2)
    x = np.random.normal(0, 1, (2, 4, 8))

    # Execute
    output = model.forward(x)

    # Verify
    expected_shape = (2, 2, 4, 4)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
```

### Performance Considerations

1. **Memory Layout Matters**: Different layouts have significant performance impact
2. **Block Size Tuning**: Default 64 aligns with typical cache lines
3. **Sequence Length**: Longer sequences benefit more from blocking
4. **Batch Processing**: Batching improves cache efficiency

### Common Tasks

#### Adding a New Memory Layout Strategy

1. Update `MemoryLayout` dataclass in `src/model/attention.py`
2. Modify `CacheAwareAttention._initialize_weights()` and `forward()`
3. Add tests in `tests/test_attention.py`
4. Update benchmarking suite to include new layout
5. Document in README.md

#### Implementing a New Attention Variant

1. Create new class inheriting from or similar to `CacheAwareAttention`
2. Implement `forward()` method with cache-aware operations
3. Add comprehensive unit tests
4. Benchmark against existing implementations
5. Document performance characteristics

#### Adding New Performance Metrics

1. Update `CacheStats` dataclass in `src/profiling/cache_monitor.py`
2. Modify platform-specific counters (`WindowsPerfCounter` or `LinuxPerfCounter`)
3. Update `_parse_perf_output()` for new metrics
4. Add visualization to `benchmark_layouts.py`

## Dependencies and Compatibility

### Required Dependencies

```
numpy>=1.21.0      # Array operations and matrix math
psutil>=5.8.0      # Process and system monitoring
pytest>=6.2.5      # Testing framework
torch>=1.9.0       # PyTorch (currently unused, prepared for future)
matplotlib>=3.4.3  # Plotting and visualization
pandas>=1.3.0      # Data manipulation (currently unused)
```

### Python Version

- **Minimum**: Python 3.7
- **Recommended**: Python 3.8+
- **Platform**: Linux (primary), Windows (supported)

### Platform-Specific Notes

**Linux**:
- Requires `perf` tool for detailed cache statistics
- May need sudo permissions for hardware counter access
- Install: `sudo apt-get install linux-tools-common linux-tools-generic`

**Windows**:
- Uses Performance Data Helper (PDH) API
- No special permissions typically required
- Metrics may be less detailed than Linux perf

## Critical Implementation Details

### Attention Mechanism

1. **Scaled Dot-Product**: Uses `sqrt(head_dim)` scaling factor
2. **Numerical Stability**: Subtracts max before softmax to prevent overflow
3. **Blocking Strategy**: Processes attention in cache-aligned blocks when enabled
4. **Head Splitting**: Reshapes and transposes to `(batch, heads, seq_len, head_dim)`

### Memory Layout Impact

From benchmark results, key findings:
- **Cache alignment** can provide 2-3x speedup for certain configurations
- **Block size of 64** aligns well with typical CPU cache lines
- **Separated QKV** sometimes outperforms contiguous storage
- **Performance varies** significantly based on batch size and sequence length

### Error Handling

1. **Dimension Validation**: Check `dim % num_heads == 0` in constructor
2. **Input Shape Validation**: Verify input dimensions match model configuration
3. **Graceful Degradation**: Fallback to standard computation if optimizations fail

## Git Workflow

### Branch Naming

- Feature branches: `claude/claude-md-<session-id>`
- Development on designated branches only
- Never push to main/master without permission

### Commit Messages

Format: `<type>: <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `test`: Add or update tests
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `bench`: Benchmark updates

Example: `feat: add flash attention memory layout`

### Making Changes

1. Create/switch to designated branch
2. Make changes with clear, focused commits
3. Run tests: `pytest tests/`
4. Update documentation if needed
5. Push to designated branch: `git push -u origin <branch-name>`

## Debugging Tips

### Common Issues

1. **Dimension Mismatch Errors**
   - Verify `dim % num_heads == 0`
   - Check input shape matches model dimension
   - Ensure batch dimension is present

2. **Performance Counter Failures**
   - Linux: Check if `perf` is installed and accessible
   - Windows: Verify PDH service is running
   - Fallback: Use basic timing without cache metrics

3. **Memory Issues**
   - Large dimensions may exceed available RAM
   - Use smaller batch sizes for testing
   - Enable blocking for long sequences

### Verification Steps

```python
# Quick sanity check
model = CacheAwareAttention(dim=8, num_heads=2)
x = np.random.randn(1, 4, 8)
out = model.forward(x)
assert not np.any(np.isnan(out))
assert out.shape == (1, 2, 4, 4)
print("✓ Basic functionality verified")
```

## Future Development Areas

Based on README.md and current implementation:

1. **PyTorch Integration**: Currently uses NumPy; PyTorch dependency prepared
2. **Multi-GPU Support**: Cache synchronization across GPUs
3. **Flash Attention**: Implement flash attention variant
4. **Dynamic Optimization**: Runtime selection of optimal layout
5. **Advanced Prefetching**: Predictive cache loading strategies
6. **Custom Cache Policies**: Beyond LRU/FIFO/LFU

## Contact and Support

- **Author**: Venkata Anudeep Adiraju
- **Email**: venkataanudeep.adiraju@utsa.edu
- **GitHub**: [@anudeepadi](https://github.com/anudeepadi)
- **Issues**: https://github.com/anudeepadi/Cache_OPT_Transformers/issues

## Quick Reference Commands

```bash
# Development setup
pip install -e .

# Run tests
pytest tests/ -v

# Run benchmarks
python benchmarks/benchmark_layouts.py

# Check code style
flake8 src/ tests/

# Git operations
git checkout -b claude/claude-md-<session-id>
git add .
git commit -m "feat: description"
git push -u origin claude/claude-md-<session-id>
```

---

**Last Updated**: 2025-11-23
**Repository Version**: 0.1.0
**Claude SDK Optimized**: Yes
