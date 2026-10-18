# Actor-state ML compiler negative-source fixtures

These are deliberately invalid `.ores` inputs. Their expected diagnostic classes
are **proposals**, not compiler-verified names. Both Java and LLVM must reject
them during source compilation before any guest execution; no host-only syntax
scan substitutes for this gate.

| Fixture | Expected diagnostic class | Prohibited behavior |
| --- | --- | --- |
| `actor_ml_raw_field_reject.ores` | `ACTOR_PERSISTENT_FIELD_FORBIDDEN` | Actor-specific `val` outside sole `state` |
| `actor_ml_state_escape_reject.ores` | `ACTOR_STATE_LOAN_ESCAPE` | Return or capture ephemeral state binder |
| `actor_ml_read_mutation_reject.ores` | `ACTOR_STATE_READONLY_WRITE` | Mutate state through `using state.read` |
| `actor_ml_missing_effect_reject.ores` | `ACTOR_EFFECT_EXCEEDS_WITH` | Declare read-only while writing |
| `actor_ml_write_missing_critical_reject.ores` | `ACTOR_STATE_WRITE_NEEDS_CRITICAL` | Write without exclusive admission |
| `actor_ml_await_loan_reject.ores` | `ACTOR_STATE_LOAN_SUSPEND` | Suspend while a state loan is live |

The normative static checker should reject mutable aliases, including aliases
buried inside a returned struct/container, not merely direct `return s`.
Compile diagnostics, guest-runtime tests, actual GC/ownership safety and the
reader/writer gate are separate evidence requirements. Neither `with` alone
nor a serial actor creates authority to read/write state. In particular, the
write effect is not inferred from an actor's execution-thread count.

Source fixture paths under `tests/fixtures/actor-state-ml-negatives` follow
this document. An external test harness should run each fixture with the
specific pinned backend revision and require failure **at the expected
compiler diagnostic** rather than a generic crash or missing import. Unusable
LLVM syntax may be recorded as an unsupported-subset diagnostic, not a pass
for full backend parity.
