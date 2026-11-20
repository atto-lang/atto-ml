# oreslang-ml: proposed integration contract v1

**Status: CONTRACT_ONLY_NOT_GUEST_EXECUTED.** The versioned machine-readable acceptance contract is [`integration/contract.v1.json`](../integration/contract.v1.json). This is an interface target, not a working implementation.

## Required integration

- Scope: linear CPU layer, MSE loss, SGD update with deterministic state ownership
- Expected upstreams: oreslang-tensor + oreslang-autograd + common-source
- Independent numeric fixture IDs: ml/linear-sgd/v1
- Guest negative evidence must include type/shape mismatch, numeric overflow, unsupported host/device, and ownership/alias violations when applicable.
- Hosted JavaScript/Python numeric oracles are independent *reference data*, never compiler/runtime parity evidence. Host-only [funded Actions #39093380746](https://github.com/scintilla-run-test/gleam-runner-conformance/actions/runs/39093380746) passed 26/26 synthetic/reference tests, but executed **no Oreslang**.
- Any actual Oreslang compiler guest pass must be bound to the immutable compiler SHA, exact source unit IDs, toolchain/target, fixture blob digests, stdout/output and negative test oracle, and actual runner steps.
- Source exposure incident [platform-conformance #22](https://github.com/ores-truffle-oreslang/oreslang-platform-conformance/issues/22) remains open. No cross-organization export of private Git source is authorized by this contract.

## Next implementation

Implement Oreslang-native layers/loss/optimizer and execute loss-decrease guest training fixture.

Run `npm test` to validate contract metadata (not Oreslang compilation). No attempt should mark this integration complete based only on a matching contract ID.
