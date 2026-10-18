# Oreslang ML

Oreslang-native neural network layers, losses, optimizers and training loops on top of `oreslang-tensor` and `oreslang-autograd`.

## P0 acceptance
- CPU-only reference model, linear forward/loss/SGD and a loss-decreasing training example.
- No hidden Python, Java or vendor ML framework implementation.
- Data and optimizer state have explicit ownership and lifecycle.

See `docs/CONTRACT.md` after bootstrap PR.
