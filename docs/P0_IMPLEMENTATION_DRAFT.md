# Oreslang ML P0: honest minimal CPU training baseline

Import `Linear` from `src/linear.ores` for the CPU reference kernels.
Read-only operands use `rt borrow`. Pass caller-owned output and parameter
buffers with `rt mut borrow`. Each updating kernel validates the full result
before writing, including when the last candidate update overflows.

The module implements affine inference, mean-squared error, one-feature
analytic-gradient SGD, and vector SGD in pure `.ores`. No Java, Python, or
TensorFlow/PyTorch implementation is called.

`tests/smoke.ores` trains `y=2*x+1` from zero-initialized parameters on a
four-example batch and requires MSE < 1e-6 after 500 full-batch SGD steps.
It also checks shape errors leave parameters and outputs unchanged.

The independent sibling-workspace examples now include:
- `examples/train_with_autograd.ores`: scalar reverse-mode training for an affine model.
- `examples/train_two_layer_autograd.ores`: scalar `Linear -> ReLU -> Linear` MLP,
  five fixed samples, 500 SGD epochs and MSE < 1e-6 target. The reference
  numerical oracle converges from MSE 3.2 to about 3.75e-7.
- `scripts/test-workspace.sh` requires a real Oreslang binary and sibling
  checkout; both example programs must finish successfully.

**Boundary:** `src/linear.ores` remains an independent analytic-gradient
baseline, while the two new examples are **scalar-autograd** training. Neither
is a generic `Tensor<T>` model nor evidence of source compiler execution.
The generic tensor-backed `Module/Linear/Sequential` and optimizer architecture
remains an explicit separate release gate.

Pending dependencies:
1. Compiler-supported unique/frozen tensor buffers, checkable view lifetimes.
2. Tensor-valued reverse-mode VJPs and leaf parameter gradients.
3. `Linear`, `Sequential`, `MSELoss`, and SGD abstraction over tensor
   parameters, with explicit parameter ownership and optimizer state.
4. Native Java/Graal execution, then actual LLVM AOT execution and parity.
5. Numeric exceptional-value and cross-backend policy. Current scalar reference now rejects nonfinite inputs/parameters and preflights candidate result values before mutating outputs, but does not prove IEEE semantics/atomic publication under concurrent aliases.

Source-level CPU kernels must not be reimplemented using hidden host
ML frameworks. Independent Python/PyTorch benchmarks are acceptable **oracles**,
not production dependencies.
