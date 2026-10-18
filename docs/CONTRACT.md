# Oreslang ML P0: honest minimal CPU training baseline

`src/linear.ores` implements affine inference, mean-squared error, one-feature
analytic-gradient SGD, and vector SGD in pure `.ores`. No Java, Python, or
TensorFlow/PyTorch implementation is called.

`tests/smoke.ores` trains `y=2*x+1` from zero-initialized parameters on a
four-example batch and requires MSE < 1e-6 after 500 full-batch SGD steps.
It also checks shape errors leave parameters and outputs unchanged.

**Boundary:** This is a **manual analytic-gradient baseline**, not
autograd-backed neural-network training or a `Tensor<T>` model. It proves
the training-loop numerical contract and provides independent reference behavior
for the future `oreslang-tensor` + `oreslang-autograd` integration.

Pending dependencies:
1. Compiler-supported unique/frozen tensor buffers, checkable view lifetimes.
2. Tensor-valued reverse-mode VJPs and leaf parameter gradients.
3. `Linear`, `Sequential`, `MSELoss`, and SGD abstraction over tensor
   parameters, with explicit parameter ownership and optimizer state.
4. Native Java/Graal execution, then actual LLVM AOT execution and parity.
5. NaN/Inf/numeric overflow policy; current scalar reference only validates
   learning-rate finiteness and input shapes.

Source-level CPU kernels must not be reimplemented using hidden host
ML frameworks. Independent Python/PyTorch benchmarks are acceptable **oracles**,
not production dependencies.
