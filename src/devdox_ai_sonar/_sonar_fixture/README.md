# SonarCloud throwaway fixtures

Throwaway Python modules for verifying `devdox_ai_sonar`'s end-to-end
fix path. Each module is shaped so SonarCloud reports a known set of
Python rule findings; the CLI fix path is then expected to call
`fix_at_line` to apply concrete edits.

Delete this directory once the verification round is complete.

---

## Rule routing reference

Source of truth: `src/devdox_ai_sonar/services/rule_handler.py` (the
`RuleHandlerRegistry` at lines 1529-1577 dispatches in the order below;
the first handler whose `can_handle` returns true wins).

| Rule(s) | Handler | Defined at | Prompt template |
|---|---|---|---|
| `python:S7503` | AsyncToSyncHandler | rule_handler.py:1245 | default agent prompt |
| `python:S3776` | CognitiveComplexityHandler | rule_handler.py:1427 | **rewritten** `prompts/python/refactoring/system_fix_issues.j2` + `user_prompt.j2` |
| `python:S117`, `python:S1172`, `python:S1542` | ConvenationNameHandler | rule_handler.py:237 | default agent prompt |
| `python:S1192` | StringLiteralDuplicateHandler | rule_handler.py:787 | default agent prompt |
| anything else (e.g. S1481, S125, S1186, S1135) | DefaultRuleHandler | rule_handler.py:1479 | default agent prompt |

The S3776 prompt swap happens at `llm_fixer.py:1506-1518`; only that
rule gets the rewritten templates. Every other rule still goes through
`prompts/python/system_agent_fix_issues.j2` + `user_agent_prompt.j2`,
which were **not** updated in the recent prompt-rewrite round.

---

## Known limitation: event-stream logs are silently dropped

`src/devdox_ai_sonar/llm_fixer.py:320` calls `_handle_event_logging(event)`,
but no such function is defined. As a result, the per-event diagnostic
log lines (`ActionEvent`, `ObservationEvent`, agent `MessageEvent`)
will not surface during a run. The remaining diagnostic logs (prompt
rendering, executor invocation, conversation finished, return) are not
in the event callback and **do** surface.

Implication for these fixtures: you can still tell whether the agent
called `fix_at_line` (the `FixAtLineExecutor INVOKED` line in
`openhands_tools/fix_at_line/impl.py:37` fires when and only when the
tool is invoked), but you cannot see the model's per-step reasoning
through the diagnostic stream.

---

## Diagnostic log checklist (per fix attempt)

Tag: `[fix_at_line-diagnostic]`. Default level: INFO. The CLI's
`--verbose` flag controls traceback printing only, not log level
(`cli.py:657, 741`); INFO logs surface on the root logger.

| # | Where | Fires when |
|---|---|---|
| 1 | llm_fixer.py:1514 | S3776 path picks the refactoring prompt (S3776 only) |
| 2 | llm_fixer.py:1114 | System prompt rendered (first 500 chars) |
| 3 | llm_fixer.py:1118 | User prompt rendered (first 500 chars) |
| 4 | llm_fixer.py:1141 | Agent run starting (model + tool list) |
| 5 | impl.py:37 | `FixAtLineExecutor INVOKED` — fires once per tool call |
| 6 | impl.py:47 | `FixAtLineExecutor RESULT` — paired with each #5 |
| 7 | llm_fixer.py:1169 | Conversation finished, includes `fix_at_line_called` flag |
| 8 | llm_fixer.py:1199 | Returning `SonarFixResponse` (success path) |
| — | llm_fixer.py:1177 | Narration warning, return `None` (failure path) |

**Healthy run:** 1 (only on S3776), 2, 3, 4, then 5+6 at least once,
then 7 with `fix_at_line_called=True`, then 8.

**Failure modes:**
- Logs 1-4, 7, then narration warning at line 1177, no 5/6/8: agent
  refused to call the tool. This is the "narrate instead of tool-call"
  failure the recent prompt rewrite was meant to fix.
- 5 fires but 6 reports `is_error=True`: the tool was called but the
  edit failed (usually `old_block` mismatch). The agent should retry;
  if it doesn't, the loop ends without applying the fix.

---

## Fixtures

### s3776_cognitive_complexity.py
- **Targets:** `python:S3776`
- **Expected SonarCloud findings:** 1, on `categorize_response`
- **Handler:** CognitiveComplexityHandler
- **Prompt path:** **rewritten** (`refactoring/*`)
- **Expected agent behavior:** `view` once, then >= 2 `fix_at_line`
  calls (one to replace `categorize_response` with a simplified body
  that calls helpers, plus one per helper to insert it as a sibling),
  then `terminal` to run `python -m py_compile ... && echo
  "SYNTAX_OK"`.
- **Pass criterion:** Logs 1 -> 8 fire, file post-edit has cognitive
  complexity < 15, helpers placed as siblings, original signature
  preserved, behaviour unchanged on a small spot-check.

### s1192_string_literals.py
- **Targets:** `python:S1192`
- **Expected SonarCloud findings:** 3 (one per duplicated literal:
  `'application/json'`, `'Outbound request started'`, `'Outbound
  request failed'`)
- **Handler:** StringLiteralDuplicateHandler
- **Prompt path:** **default** (not rewritten)
- **Expected agent behavior:** for each literal, extract a
  module-level constant and replace each occurrence with the
  constant. In the rewritten-prompt model that means three
  `fix_at_line` runs (one to insert the constant, one or more to
  replace each occurrence). Under the default prompt the agent has
  historically returned JSON instead of calling the tool.
- **Pass criterion:** post-edit, each literal appears at most once
  (in the constant assignment), and a follow-up SonarCloud scan
  reports zero S1192 findings on this file.
- **This is the cleanest "is the default prompt broken?" probe.**
  If S3776 succeeds and this fails with a narration warning, that
  confirms the prompt-rewrite hypothesis and tells you the default
  agent prompt also needs the same treatment.

### s117_naming.py
- **Targets:** `python:S117` (variable / parameter naming),
  `python:S1172` (unused parameter), `python:S1542` (function name)
- **Expected SonarCloud findings:** several S117 (one per offending
  identifier), 1 S1172 (`MultiplierUnused`), 1 S1542 (`ProcessRecord`)
- **Handler:** ConvenationNameHandler
- **Prompt path:** default
- **Expected agent behavior:** rename in place via `fix_at_line`
  per identifier (and per call site).
- **Pass criterion:** post-edit, every parameter / local / function
  name matches `^[_a-z][_a-z0-9]*$`; the unused parameter is removed
  or marked underscored; no callers exist within this file so rename
  cannot cascade across modules.

### s7503_async_to_sync.py
- **Targets:** `python:S7503` (async function with no await)
- **Expected SonarCloud findings:** 3 (one per async function)
- **Handler:** AsyncToSyncHandler
- **Prompt path:** default
- **Expected agent behavior:** drop the `async` keyword from each
  function and adjust any returns. No callers exist within this file.
- **Pass criterion:** post-edit, all three functions are `def`, not
  `async def`; no `await` was needed.

### default_handler_grab_bag.py
- **Targets:** `python:S1481`, `python:S125`, `python:S1186`,
  `python:S1135`
- **Expected SonarCloud findings:**
  - S1481 on `count` in `compute_average`
  - S125 on the commented block in `render_user`
  - S1186 on `placeholder`
  - S1135 on the TODO comment in `settle_invoice`
- **Handler:** DefaultRuleHandler (catch-all)
- **Prompt path:** default
- **Expected agent behavior:** delete the unused local; remove the
  commented-out block; either implement `placeholder` or remove it
  (rules of thumb here are vague — agent may hand-wave); resolve or
  remove the TODO.
- **Pass criterion:** post-edit, follow-up SonarCloud scan reports
  zero of these four rules on this file. Note that S1186 may be
  ambiguous to fix without context.

### mixed_multi_rule.py
- **Targets:** `python:S3776` + `python:S1192` + `python:S1481` +
  `python:S125`, all in one file
- **Expected SonarCloud findings:**
  - S3776 on `route_request`
  - S1192 on `'application/json'`
  - S1481 on `unused_total`
  - S125 on the commented block in `archive_record`
- **Handlers exercised:** CognitiveComplexityHandler,
  StringLiteralDuplicateHandler, DefaultRuleHandler — all on the
  same file in a single CLI run
- **Prompt paths exercised:** rewritten (S3776) **and** default
  (S1192, S1481, S125)
- **What this stresses:** line-offset stability after edits. The
  first fix changes the file's line numbers; subsequent fixes
  receive `start_line` / `end_line` from SonarCloud findings that
  were computed against the **pre-edit** file. Watch log #6
  (`FixAtLineExecutor RESULT`) for `is_error=True` with `old_block`
  mismatch messages — that is the canonical drift symptom.
- **Pass criterion:** post-edit, follow-up SonarCloud scan reports
  zero findings of any of the four rules on this file, and the file
  still parses (`python -m py_compile`).

---

## Pre-existing combined fixture

`temp_test_module.py` (in this directory) is the original throwaway
target. It carries the same shape as `s3776_cognitive_complexity.py`
+ `s1192_string_literals.py` rolled into one file. Keep or delete at
your discretion; it does not duplicate the cleaner single-rule
fixtures here, but it does have broader noise.

---

## Running the verification round

The CLI is what you'd normally run to fix issues for the project /
branch this directory lives in. With these fixtures committed and
pushed (so SonarCloud picks them up on its next scan), trigger a
fresh scan, then run the fix command. Watch stdout / log file for
`[fix_at_line-diagnostic]` lines and cross-check against the
checklist above.

After each run, before re-running, check `git diff` on the fixture
files to see what the agent actually applied. If a fix was wrong but
syntactically clean, restore the file with `git checkout --` on the
specific path and adjust the fixture or the prompt before retrying.
