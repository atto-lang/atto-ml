# PR #16 MFKAN dependency-aware reconciliation

Historical head `267ee1c0b7e0fc43f751499faa09058506e20b07`; latest stack parent #15 `af208d76d777887ef5179331217ee38a5dca8fc4`. Do NOT rebase MFKAN onto raw main: it imports the cubic spline module owned by the still-unmerged parent.

Kept parent #15's input derivative oracle and run/check hooks, pinned spline source and numerical constraints. Forward-ported MFKAN source, independent host numerical tests, pinned guest smoke/verifier and spec. Merged workflow runs both parent VJP and child MFKAN tests. Guest remains NOT_EXECUTED without independently pinned compiler/source evidence. Require full exact-head Actions and source execution; PR #15 must land before #16.
