# Oreslang ML

Oreslang-native neural network layers, losses, optimizers and training loops on top of `oreslang-tensor` and `oreslang-autograd`.

**Current status:** `main` is contract/metadata-only; native ML examples and
kernels are under review on stacked draft PRs. There is no compiled-guest
Java/LLVM release proof. See [security and release gates](docs/ML_SECURITY_RELEASE_GATES.md)
before using any prototype for production or claiming backend parity.

## P0 acceptance
- CPU-only reference model, linear forward/loss/SGD and a loss-decreasing training example.
- No hidden Python, Java or vendor ML framework implementation.
- Data and optimizer state have explicit ownership and lifecycle.

See `docs/CONTRACT.md` after bootstrap PR.
