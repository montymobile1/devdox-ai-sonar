"""
S3776 Cognitive Complexity — Debugging & Regression Module
==========================================================

PURPOSE:
    Throwaway module designed to trigger ~25 python:S3776 violations in SonarCloud.
    Each complex function uses contrived nesting to exceed the cognitive complexity
    threshold (default 15). Simple helper functions are interspersed to test
    selective fixing, call-dependency preservation, and structural integrity.

WHAT TO EXPECT FROM SONARCLOUD:
    - ~25 S3776 violations (one per complex function marked COMPLEX below)
    - Expected line ranges for each function are documented in comments above each

WHAT TO VALIDATE DURING FIX (checklist):

  LINE RANGE ACCURACY:
  [ ] 1.  FixSuggestion.line_number vs actual function start — do they match?
  [ ] 2.  FixSuggestion.last_line_number vs actual function end — is it too short?
         (Expected bug: last_line = last flow contributor, not function end)
  [ ] 3.  CodeBlock.start_line / end_line — do they cover the full function?
  [ ] 4.  Multi-line signature: does SonarCloud report line of `def` or last param?
  [ ] 5.  Decorated functions: does start_line include decorator or just `def`?

  FIX APPLICATION:
  [ ] 6.  apply_single_fix return value — success=True or False? If False, what reason?
  [ ] 7.  check_python_interpreter — does "python" binary exist on PATH? (vs python3)
  [ ] 8.  After fix: does actual source file change, or only .tmp.py?
  [ ] 9.  LineRange.is_valid — does validation pass with possibly-wrong last_line?

  STRUCTURAL INTEGRITY:
  [ ] 10. Adjacent violations (#4, #5, #6) — all fixed independently without overlap?
  [ ] 11. Cross-call dependencies — after refactoring #9, does it still correctly call #4?
  [ ] 12. Chain calls — after refactoring #40→#41→#42, do call chains survive?
  [ ] 13. Mutual recursion — after refactoring #13 and #14, do cross-calls survive?
  [ ] 14. Recursive function — does self-call survive refactoring?
  [ ] 15. Shared helper — is #7 (simple_shared_helper) untouched after fixing #4, #5, #6?

  CLASS/INDENTATION:
  [ ] 16. Class method fix — is indentation preserved? (4-space body → 8-space?)
  [ ] 17. @staticmethod preserved above refactored code?
  [ ] 18. @classmethod preserved?
  [ ] 19. Stacked decorators — all preserved in correct order?
  [ ] 20. Nested class method — double indentation preserved?

  CONTENT PRESERVATION:
  [ ] 21. Docstring preserved after fix?
  [ ] 22. Inline comments preserved?
  [ ] 23. yield keyword preserved (generator)?
  [ ] 24. async/await preserved?
  [ ] 25. Type annotations in signature preserved?
  [ ] 26. Nested function definition preserved or extracted?
  [ ] 27. try/except/finally structure maintained?
  [ ] 28. Context managers (with) maintained?

  EDGE CASES:
  [ ] 29. EOF function — replacement handles end-of-file correctly?
  [ ] 30. First-in-file function — no off-by-one at line 1?
  [ ] 31. Very large function (~60 lines) — full replacement works?
  [ ] 32. Barely-over-threshold — does minimal violation get fixed cleanly?
  [ ] 33. Between-classes function — file structure preserved?
  [ ] 34. Helper code placement — SIBLING/GLOBAL_TOP correct?

FUNCTION LAYOUT:
    See individual function comments for case IDs [A1]-[F2].
    Total: ~25 COMPLEX, ~18 simple, ~43 functions.
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


# --------------------------------------------------------------------------
# 1. complex_top_of_file — COMPLEX [A1]
#    First function in the file after imports.
#    Tests: start_line accuracy at file beginning.
#    Expected SonarCloud line: line of `def`
# --------------------------------------------------------------------------
def complex_top_of_file(data: dict) -> Optional[str]:
    result = None
    if data:  # +1
        for key in data:  # +1 (nesting=1) +1
            if key.startswith("a"):  # +1 (nesting=2) +2
                if data[key] > 10:  # +1 (nesting=3) +3
                    if isinstance(data[key], int):  # +1 (nesting=4) +4
                        result = f"found-{key}"
                    else:  # +1
                        result = "non-int"
                elif data[key] < 0:  # +1
                    for sub in range(abs(data[key])):  # +1 (nesting=4) +4
                        if sub % 2 == 0:  # +1 (nesting=5) +5
                            result = str(sub)
            elif key.startswith("b"):  # +1
                if data[key] is None:  # +1 (nesting=3) +3
                    result = "none-b"
                else:  # +1
                    result = str(data[key])
    simple_helper_a(result)
    return result


# --------------------------------------------------------------------------
# 2. simple_helper_a — simple (called by #1)
# --------------------------------------------------------------------------
def simple_helper_a(value: Optional[str]) -> str:
    if value is None:
        return "default"
    return value.upper()


# --------------------------------------------------------------------------
# 3. simple_pure_b — simple (standalone)
# --------------------------------------------------------------------------
def simple_pure_b(x: int) -> int:
    return x * 2 + 1


# --------------------------------------------------------------------------
# 4. complex_adjacent_alpha — COMPLEX [A3]
#    First of back-to-back complex functions.
#    Tests: adjacent violation handling, no bleed-over.
# --------------------------------------------------------------------------
def complex_adjacent_alpha(items: list) -> int:
    total = 0
    if items:  # +1
        for item in items:  # +1 (nesting=1) +1
            if isinstance(item, dict):  # +1 (nesting=2) +2
                for k, v in item.items():  # +1 (nesting=3) +3
                    if v > 0:  # +1 (nesting=4) +4
                        total += v
                    elif v < 0:  # +1
                        total -= v
                    else:  # +1
                        total += 1
            elif isinstance(item, list):  # +1
                for sub in item:  # +1 (nesting=3) +3
                    if sub:  # +1 (nesting=4) +4
                        total += sub
            elif isinstance(item, int):  # +1
                if item > 100:  # +1 (nesting=3) +3
                    total += item
    simple_shared_helper(total)
    return total


# --------------------------------------------------------------------------
# 5. complex_adjacent_beta — COMPLEX [A3]
#    Second of back-to-back complex functions. Calls #4.
#    Tests: adjacent violation handling.
# --------------------------------------------------------------------------
def complex_adjacent_beta(records: list) -> list:
    results = []
    if records:  # +1
        for rec in records:  # +1 (nesting=1) +1
            if "type" in rec:  # +1 (nesting=2) +2
                if rec["type"] == "alpha":  # +1 (nesting=3) +3
                    val = complex_adjacent_alpha(rec.get("items", []))
                    if val > 50:  # +1 (nesting=4) +4
                        results.append(val)
                    else:  # +1
                        results.append(0)
                elif rec["type"] == "beta":  # +1
                    if rec.get("active"):  # +1 (nesting=4) +4
                        for sub in rec.get("subs", []):  # +1 (nesting=5) +5
                            if sub > 0:  # +1 (nesting=6) +6
                                results.append(sub)
                elif rec["type"] == "gamma":  # +1
                    results.append(-1)
            else:  # +1
                results.append(None)
    simple_shared_helper(len(results))
    return results


# --------------------------------------------------------------------------
# 6. complex_adjacent_gamma — COMPLEX [A4]
#    Third in a row of back-to-back complex functions.
#    Tests: stress test 3 consecutive violations.
# --------------------------------------------------------------------------
def complex_adjacent_gamma(matrix: list) -> dict:
    stats = {"pos": 0, "neg": 0, "zero": 0}
    if matrix:  # +1
        for row in matrix:  # +1 (nesting=1) +1
            if isinstance(row, list):  # +1 (nesting=2) +2
                for val in row:  # +1 (nesting=3) +3
                    if val > 0:  # +1 (nesting=4) +4
                        stats["pos"] += 1
                        if val > 100:  # +1 (nesting=5) +5
                            stats["pos"] += val
                    elif val < 0:  # +1
                        stats["neg"] += 1
                        if val < -100:  # +1 (nesting=5) +5
                            stats["neg"] += abs(val)
                    else:  # +1
                        stats["zero"] += 1
            elif isinstance(row, int):  # +1
                if row > 0:  # +1 (nesting=3) +3
                    stats["pos"] += row
                else:  # +1
                    stats["neg"] += abs(row)
    simple_shared_helper(sum(stats.values()))
    return stats


# --------------------------------------------------------------------------
# 7. simple_shared_helper — simple [B3]
#    Called by #4, #5, #6. Must survive multiple refactoring passes.
# --------------------------------------------------------------------------
def simple_shared_helper(count: int) -> None:
    logger.debug("Processed %d items", count)


# --------------------------------------------------------------------------
# 8. simple_pure_c — simple
# --------------------------------------------------------------------------
def simple_pure_c(s: str) -> str:
    return s.strip().lower()


# --------------------------------------------------------------------------
# 9. complex_calls_complex — COMPLEX [B1]
#    Calls another complex function (#4).
#    Tests: does refactoring one break the other's call?
# --------------------------------------------------------------------------
def complex_calls_complex(datasets: list) -> int:
    grand_total = 0
    if datasets:  # +1
        for ds in datasets:  # +1 (nesting=1) +1
            if isinstance(ds, dict):  # +1 (nesting=2) +2
                items = ds.get("items", [])
                if items:  # +1 (nesting=3) +3
                    partial = complex_adjacent_alpha(items)
                    if partial > 100:  # +1 (nesting=4) +4
                        grand_total += partial
                    elif partial > 50:  # +1
                        grand_total += partial // 2
                    else:  # +1
                        grand_total += 1
                else:  # +1
                    if ds.get("default"):  # +1 (nesting=4) +4
                        grand_total += ds["default"]
            elif isinstance(ds, list):  # +1
                for item in ds:  # +1 (nesting=3) +3
                    if item:  # +1 (nesting=4) +4
                        grand_total += item
    return grand_total


# --------------------------------------------------------------------------
# 10. complex_calls_simple — COMPLEX [B2]
#     Calls simple helpers #7 and #8.
#     Tests: are simple helpers left untouched?
# --------------------------------------------------------------------------
def complex_calls_simple(entries: list) -> list:
    output = []
    if entries:  # +1
        for entry in entries:  # +1 (nesting=1) +1
            if isinstance(entry, str):  # +1 (nesting=2) +2
                cleaned = simple_pure_c(entry)
                if cleaned:  # +1 (nesting=3) +3
                    if cleaned.startswith("x"):  # +1 (nesting=4) +4
                        output.append(cleaned)
                    elif cleaned.startswith("y"):  # +1
                        output.append(cleaned[::-1])
                    else:  # +1
                        output.append(cleaned.upper())
            elif isinstance(entry, int):  # +1
                if entry > 0:  # +1 (nesting=3) +3
                    for i in range(entry):  # +1 (nesting=4) +4
                        if i % 3 == 0:  # +1 (nesting=5) +5
                            output.append(str(i))
            else:  # +1
                output.append(None)
    simple_shared_helper(len(output))
    return output


# --------------------------------------------------------------------------
# 11. simple_pure_d — simple
# --------------------------------------------------------------------------
def simple_pure_d(a: int, b: int) -> int:
    return a + b


# --------------------------------------------------------------------------
# 12. complex_recursive — COMPLEX [B4]
#     Calls itself recursively.
#     Tests: does the self-call survive refactoring?
# --------------------------------------------------------------------------
def complex_recursive(tree: dict, depth: int = 0) -> int:
    count = 0
    if not tree:  # +1
        return 0
    if "value" in tree:  # +1
        if tree["value"] > 0:  # +1 (nesting=1) +1
            count += tree["value"]
            if tree["value"] > 100:  # +1 (nesting=2) +2
                count += tree["value"] // 10
        elif tree["value"] < 0:  # +1
            count -= 1
    if "children" in tree:  # +1
        for child in tree["children"]:  # +1 (nesting=1) +1
            if isinstance(child, dict):  # +1 (nesting=2) +2
                child_count = complex_recursive(child, depth + 1)
                if child_count > 0:  # +1 (nesting=3) +3
                    count += child_count
                    if depth > 5:  # +1 (nesting=4) +4
                        count += depth
            elif isinstance(child, int):  # +1
                if child > 0:  # +1 (nesting=3) +3
                    count += child
    return count


# --------------------------------------------------------------------------
# 13. complex_mutual_a — COMPLEX [B6]
#     Calls #14 (mutual recursion).
#     Tests: circular dependency handling.
# --------------------------------------------------------------------------
def complex_mutual_a(nodes: list, visited: Optional[set] = None) -> int:
    if visited is None:
        visited = set()
    total = 0
    if nodes:  # +1
        for i, node in enumerate(nodes):  # +1 (nesting=1) +1
            if i in visited:  # +1 (nesting=2) +2
                continue  # already seen
            visited.add(i)
            if isinstance(node, dict):  # +1 (nesting=2) +2
                if node.get("delegate"):  # +1 (nesting=3) +3
                    total += complex_mutual_b(node.get("sub_nodes", []), visited)
                elif node.get("value"):  # +1
                    if node["value"] > 10:  # +1 (nesting=4) +4
                        total += node["value"]
                    else:  # +1
                        total += 1
            elif isinstance(node, list):  # +1
                if len(node) > 0:  # +1 (nesting=3) +3
                    for sub in node:  # +1 (nesting=4) +4
                        if sub:  # +1 (nesting=5) +5
                            total += sub
    return total


# --------------------------------------------------------------------------
# 14. complex_mutual_b — COMPLEX [B6]
#     Calls #13 (mutual recursion).
#     Tests: circular dependency handling.
# --------------------------------------------------------------------------
def complex_mutual_b(nodes: list, visited: Optional[set] = None) -> int:
    if visited is None:
        visited = set()
    total = 0
    if nodes:  # +1
        for i, node in enumerate(nodes):  # +1 (nesting=1) +1
            if i in visited:  # +1 (nesting=2) +2
                continue
            visited.add(i)
            if isinstance(node, dict):  # +1 (nesting=2) +2
                if node.get("recurse"):  # +1 (nesting=3) +3
                    total += complex_mutual_a(node.get("children", []), visited)
                elif node.get("count"):  # +1
                    if node["count"] > 5:  # +1 (nesting=4) +4
                        total += node["count"]
                    elif node["count"] > 0:  # +1
                        total += 1
                    else:  # +1
                        total -= 1
            elif isinstance(node, int):  # +1
                if node > 0:  # +1 (nesting=3) +3
                    total += node
                elif node < -10:  # +1
                    total -= 10
    return total


# --------------------------------------------------------------------------
# 15. simple_pure_e — simple
# --------------------------------------------------------------------------
def simple_pure_e(values: list) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


# ══════════════════════════════════════════════════════════════════════════════
# CLASS: DebugProcessor
# ══════════════════════════════════════════════════════════════════════════════


def some_decorator(func):
    """Trivial decorator for testing stacked decorators."""
    return func


def another_decorator(func):
    """Second trivial decorator for testing stacked decorators."""
    return func


class DebugProcessor:
    """
    Class containing methods of various complexity levels.
    Tests class context detection, self/cls handling, and indentation.
    """

    # ----------------------------------------------------------------------
    # 16. __init__ — simple [C5]
    # ----------------------------------------------------------------------
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.state = {}

    # ----------------------------------------------------------------------
    # 17. complex_method — COMPLEX [C1]
    #     Instance method.
    #     Tests: class context detection, `self` handling, indentation.
    # ----------------------------------------------------------------------
    def complex_method(self, data: list) -> dict:
        result = {}
        if data:  # +1
            for item in data:  # +1 (nesting=1) +1
                if isinstance(item, dict):  # +1 (nesting=2) +2
                    for k, v in item.items():  # +1 (nesting=3) +3
                        if k in self.config:  # +1 (nesting=4) +4
                            if v > self.config[k]:  # +1 (nesting=5) +5
                                result[k] = v
                            else:  # +1
                                result[k] = self.config[k]
                        elif k.startswith("_"):  # +1
                            self.state[k] = v
                elif isinstance(item, str):  # +1
                    if item in self.state:  # +1 (nesting=3) +3
                        result[item] = self.state[item]
                    else:  # +1
                        result[item] = None
        return result

    # ----------------------------------------------------------------------
    # 18. simple_method — simple [C5]
    # ----------------------------------------------------------------------
    def simple_method(self) -> int:
        return len(self.state)

    # ----------------------------------------------------------------------
    # 19. complex_static — COMPLEX [C2]
    #     @staticmethod.
    #     Tests: decorator + no `self`, indentation.
    # ----------------------------------------------------------------------
    @staticmethod
    def complex_static(records: list) -> list:
        filtered = []
        if records:  # +1
            for rec in records:  # +1 (nesting=1) +1
                if isinstance(rec, dict):  # +1 (nesting=2) +2
                    if rec.get("active"):  # +1 (nesting=3) +3
                        if rec.get("score", 0) > 50:  # +1 (nesting=4) +4
                            filtered.append(rec)
                        elif rec.get("score", 0) > 25:  # +1
                            if rec.get("priority") == "high":  # +1 (nesting=5) +5
                                filtered.append(rec)
                    elif rec.get("archived"):  # +1
                        if rec.get("important"):  # +1 (nesting=4) +4
                            filtered.append(rec)
                elif isinstance(rec, list):  # +1
                    for sub in rec:  # +1 (nesting=3) +3
                        if sub:  # +1 (nesting=4) +4
                            filtered.append(sub)
        return filtered

    # ----------------------------------------------------------------------
    # 20. complex_classmethod — COMPLEX [C3]
    #     @classmethod.
    #     Tests: decorator + `cls` arg, indentation.
    # ----------------------------------------------------------------------
    @classmethod
    def complex_classmethod(cls, config_data: dict) -> "DebugProcessor":
        instance = cls()
        if config_data:  # +1
            for section, values in config_data.items():  # +1 (nesting=1) +1
                if isinstance(values, dict):  # +1 (nesting=2) +2
                    for k, v in values.items():  # +1 (nesting=3) +3
                        if v is not None:  # +1 (nesting=4) +4
                            if isinstance(v, int) and v > 0:  # +1 (nesting=5) +5
                                instance.config[f"{section}.{k}"] = v
                            elif isinstance(v, str) and v:  # +1
                                instance.config[f"{section}.{k}"] = v
                            else:  # +1
                                instance.config[f"{section}.{k}"] = str(v)
                elif isinstance(values, list):  # +1
                    for idx, item in enumerate(values):  # +1 (nesting=3) +3
                        if item:  # +1 (nesting=4) +4
                            instance.config[f"{section}.{idx}"] = item
        return instance

    # ----------------------------------------------------------------------
    # 21. complex_multi_decorated — COMPLEX [C4]
    #     Multiple stacked decorators.
    #     Tests: multi-decorator detection (_find_decorator_start).
    # ----------------------------------------------------------------------
    @some_decorator
    @another_decorator
    def complex_multi_decorated(self, events: list) -> list:
        processed = []
        if events:  # +1
            for event in events:  # +1 (nesting=1) +1
                if event.get("type") == "click":  # +1 (nesting=2) +2
                    if event.get("target"):  # +1 (nesting=3) +3
                        if event["target"].startswith("btn"):  # +1 (nesting=4) +4
                            processed.append(("click", event["target"]))
                        else:  # +1
                            processed.append(("click", "unknown"))
                elif event.get("type") == "hover":  # +1
                    if event.get("duration", 0) > 1000:  # +1 (nesting=3) +3
                        processed.append(("long_hover", event.get("target")))
                    elif event.get("duration", 0) > 500:  # +1
                        processed.append(("hover", event.get("target")))
                elif event.get("type") == "scroll":  # +1
                    if event.get("distance", 0) > 500:  # +1 (nesting=3) +3
                        processed.append(("big_scroll", event.get("distance")))
        return processed

    # ------------------------------------------------------------------
    # Nested class [C6]
    # ------------------------------------------------------------------
    class InnerProcessor:
        """Nested class to test deeply nested class context."""

        # ------------------------------------------------------------------
        # 22. complex_nested_class_method — COMPLEX [C6]
        #     Method in a nested class.
        #     Tests: deeply nested class context, double indentation.
        # ------------------------------------------------------------------
        def complex_nested_class_method(self, items: list) -> dict:
            counts = {"a": 0, "b": 0, "other": 0}
            if items:  # +1
                for item in items:  # +1 (nesting=1) +1
                    if isinstance(item, str):  # +1 (nesting=2) +2
                        if item.startswith("a"):  # +1 (nesting=3) +3
                            counts["a"] += 1
                            if len(item) > 5:  # +1 (nesting=4) +4
                                counts["a"] += len(item)
                        elif item.startswith("b"):  # +1
                            counts["b"] += 1
                            if len(item) > 10:  # +1 (nesting=4) +4
                                counts["b"] += len(item)
                        else:  # +1
                            counts["other"] += 1
                    elif isinstance(item, int):  # +1
                        if item > 0:  # +1 (nesting=3) +3
                            counts["a"] += item
                        elif item < 0:  # +1
                            counts["b"] += abs(item)
                        else:  # +1
                            counts["other"] += 1
            return counts


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL (continued)
# ══════════════════════════════════════════════════════════════════════════════


# --------------------------------------------------------------------------
# 23. simple_pure_f — simple
# --------------------------------------------------------------------------
def simple_pure_f(items: list) -> list:
    return sorted(set(items))


# --------------------------------------------------------------------------
# 24. complex_multiline_sig — COMPLEX [D1]
#     Multi-line function signature.
#     Tests: does SonarCloud report line of `def` or last param line?
# --------------------------------------------------------------------------
def complex_multiline_sig(
    alpha: int,
    beta: str,
    gamma: float,
    delta: Optional[list] = None,
    epsilon: bool = False,
    zeta: Optional[dict] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if alpha > 0:  # +1
        if beta:  # +1 (nesting=1) +1
            if gamma > 1.0:  # +1 (nesting=2) +2
                result["combined"] = alpha * gamma
                if epsilon:  # +1 (nesting=3) +3
                    result["flag"] = True
                    if delta:  # +1 (nesting=4) +4
                        for item in delta:  # +1 (nesting=5) +5
                            if item:  # +1 (nesting=6) +6
                                result[str(item)] = item
            elif gamma < 0:  # +1
                result["negative"] = True
        elif zeta:  # +1
            for k, v in zeta.items():  # +1 (nesting=2) +2
                if v:  # +1 (nesting=3) +3
                    result[k] = v
    elif alpha < 0:  # +1
        result["error"] = "negative alpha"
    return result


# --------------------------------------------------------------------------
# 25. complex_async — COMPLEX [D2]
#     async def function.
#     Tests: async function handling, async/await preservation.
# --------------------------------------------------------------------------
async def complex_async(sources: list) -> list:
    results = []
    if sources:  # +1
        for src in sources:  # +1 (nesting=1) +1
            if isinstance(src, dict):  # +1 (nesting=2) +2
                if src.get("url"):  # +1 (nesting=3) +3
                    if src.get("method") == "GET":  # +1 (nesting=4) +4
                        results.append(("get", src["url"]))
                    elif src.get("method") == "POST":  # +1
                        if src.get("body"):  # +1 (nesting=5) +5
                            results.append(("post", src["url"], src["body"]))
                        else:  # +1
                            results.append(("post", src["url"], None))
                    else:  # +1
                        results.append(("other", src["url"]))
                elif src.get("path"):  # +1
                    if os.path.exists(src["path"]):  # +1 (nesting=4) +4
                        results.append(("file", src["path"]))
            elif isinstance(src, str):  # +1
                if src.startswith("http"):  # +1 (nesting=3) +3
                    results.append(("url", src))
    return results


# --------------------------------------------------------------------------
# 26. complex_generator — COMPLEX [D3]
#     Generator function (has yield).
#     Tests: generator preservation after refactoring.
# --------------------------------------------------------------------------
def complex_generator(data_stream: list) -> Any:
    if data_stream:  # +1
        for chunk in data_stream:  # +1 (nesting=1) +1
            if isinstance(chunk, dict):  # +1 (nesting=2) +2
                if chunk.get("valid"):  # +1 (nesting=3) +3
                    if chunk.get("priority") == "high":  # +1 (nesting=4) +4
                        yield {"type": "high", "data": chunk}
                    elif chunk.get("priority") == "medium":  # +1
                        if chunk.get("size", 0) < 1000:  # +1 (nesting=5) +5
                            yield {"type": "medium", "data": chunk}
                    else:  # +1
                        yield {"type": "low", "data": chunk}
                elif chunk.get("retry"):  # +1
                    for attempt in range(3):  # +1 (nesting=4) +4
                        if attempt > 0:  # +1 (nesting=5) +5
                            yield {"type": "retry", "attempt": attempt}
            elif isinstance(chunk, list):  # +1
                for item in chunk:  # +1 (nesting=3) +3
                    if item:  # +1 (nesting=4) +4
                        yield item


# --------------------------------------------------------------------------
# 27. complex_typed — COMPLEX [D4]
#     Type hints in signature.
#     Tests: type annotations survive refactoring.
# --------------------------------------------------------------------------
def complex_typed(
    x: int, y: str, z: List[Dict[str, Any]]
) -> Tuple[bool, Optional[str]]:
    found = False
    message = None
    if x > 0:  # +1
        for entry in z:  # +1 (nesting=1) +1
            if y in entry:  # +1 (nesting=2) +2
                if entry[y] > x:  # +1 (nesting=3) +3
                    found = True
                    if entry[y] > x * 10:  # +1 (nesting=4) +4
                        message = "very high"
                    elif entry[y] > x * 5:  # +1
                        message = "high"
                    else:  # +1
                        message = "moderate"
                elif entry[y] == x:  # +1
                    found = True
                    message = "exact"
            elif "default" in entry:  # +1
                if entry["default"]:  # +1 (nesting=3) +3
                    message = str(entry["default"])
    elif x < 0:  # +1
        if z:  # +1 (nesting=1) +1
            message = "negative input"
    return (found, message)


# --------------------------------------------------------------------------
# 28. simple_pure_g — simple
# --------------------------------------------------------------------------
def simple_pure_g(text: str) -> list:
    return text.split()


# --------------------------------------------------------------------------
# 29. complex_nested_def — COMPLEX [E1]
#     Contains a nested function definition.
#     Tests: inner function extraction — is nested def part of replacement?
# --------------------------------------------------------------------------
def complex_nested_def(items: list) -> list:
    def _inner_transform(val):
        if isinstance(val, str):
            return val.upper()
        return str(val)

    output = []
    if items:  # +1
        for item in items:  # +1 (nesting=1) +1
            if isinstance(item, dict):  # +1 (nesting=2) +2
                if item.get("transform"):  # +1 (nesting=3) +3
                    transformed = _inner_transform(item.get("value", ""))
                    if transformed:  # +1 (nesting=4) +4
                        output.append(transformed)
                elif item.get("skip"):  # +1
                    continue
                else:  # +1
                    if item.get("value"):  # +1 (nesting=4) +4
                        output.append(item["value"])
            elif isinstance(item, list):  # +1
                for sub in item:  # +1 (nesting=3) +3
                    val = _inner_transform(sub)
                    if val:  # +1 (nesting=4) +4
                        output.append(val)
            else:  # +1
                output.append(_inner_transform(item))
    return output


# --------------------------------------------------------------------------
# 30. complex_try_except — COMPLEX [E2]
#     try/except/finally nesting.
#     Tests: exception handling complexity.
# --------------------------------------------------------------------------
def complex_try_except(operations: list) -> dict:
    results = {"success": 0, "error": 0, "skipped": 0}
    if operations:  # +1
        for op in operations:  # +1 (nesting=1) +1
            try:  # +1
                if op.get("type") == "compute":  # +1 (nesting=3) +3
                    val = op.get("value", 0)
                    if val > 0:  # +1 (nesting=4) +4
                        result = val * 2
                        if result > 1000:  # +1 (nesting=5) +5
                            results["success"] += result
                        else:  # +1
                            results["success"] += 1
                    elif val < 0:  # +1
                        raise ValueError(f"Negative value: {val}")
                elif op.get("type") == "validate":  # +1
                    if op.get("strict"):  # +1 (nesting=4) +4
                        if not op.get("data"):  # +1 (nesting=5) +5
                            raise KeyError("Missing data")
                        results["success"] += 1
            except ValueError:  # +1
                results["error"] += 1
            except KeyError:  # +1
                results["skipped"] += 1
            finally:
                logger.debug("Operation processed")
    return results


# --------------------------------------------------------------------------
# 31. complex_context_managers — COMPLEX [E3]
#     Nested context managers (with/with).
#     Tests: context manager depth.
# --------------------------------------------------------------------------
def complex_context_managers(file_paths: list) -> dict:
    report = {"processed": 0, "errors": 0}
    if file_paths:  # +1
        for fp in file_paths:  # +1 (nesting=1) +1
            if os.path.exists(fp):  # +1 (nesting=2) +2
                try:  # +1
                    if fp.endswith(".txt"):  # +1 (nesting=4) +4
                        if os.path.getsize(fp) > 0:  # +1 (nesting=5) +5
                            report["processed"] += 1
                            if os.path.getsize(fp) > 10000:  # +1 (nesting=6) +6
                                report["processed"] += 10
                        else:  # +1
                            report["errors"] += 1
                    elif fp.endswith(".csv"):  # +1
                        report["processed"] += 1
                        if os.path.getsize(fp) > 5000:  # +1 (nesting=5) +5
                            report["processed"] += 5
                except OSError:  # +1
                    report["errors"] += 1
            else:  # +1
                report["errors"] += 1
    return report


# --------------------------------------------------------------------------
# 32. complex_comprehensions — COMPLEX [E4]
#     Comprehension complexity.
#     Tests: list/dict comprehensions.
# --------------------------------------------------------------------------
def complex_comprehensions(datasets: list) -> dict:
    output = {}
    if datasets:  # +1
        for ds in datasets:  # +1 (nesting=1) +1
            if isinstance(ds, dict):  # +1 (nesting=2) +2
                if ds.get("type") == "filter":  # +1 (nesting=3) +3
                    filtered = [
                        v for v in ds.get("values", [])
                        if v and v > 0
                    ]
                    if filtered:  # +1 (nesting=4) +4
                        output[ds.get("name", "unnamed")] = filtered
                elif ds.get("type") == "transform":  # +1
                    transformed = {
                        k: v * 2
                        for k, v in ds.get("mapping", {}).items()
                        if v is not None
                    }
                    if transformed:  # +1 (nesting=4) +4
                        output[ds.get("name", "unnamed")] = transformed
                elif ds.get("type") == "nested":  # +1
                    if ds.get("matrix"):  # +1 (nesting=4) +4
                        flat = [
                            cell
                            for row in ds["matrix"]
                            for cell in row
                            if cell
                        ]
                        if flat:  # +1 (nesting=5) +5
                            output["flat"] = flat
            elif isinstance(ds, list):  # +1
                if ds:  # +1 (nesting=3) +3
                    output["raw"] = ds
    return output


# --------------------------------------------------------------------------
# 33. barely_complex — COMPLEX [E5]
#     Complexity ~16-17 (barely over threshold of 15).
#     Tests: minimal violation detection and clean fix.
# --------------------------------------------------------------------------
def barely_complex(config: dict) -> str:
    status = "unknown"
    if config:  # +1
        if config.get("enabled"):  # +1 (nesting=1) +1
            if config.get("mode") == "strict":  # +1 (nesting=2) +2
                if config.get("level", 0) > 5:  # +1 (nesting=3) +3
                    status = "strict-high"
                else:  # +1
                    status = "strict-low"
            elif config.get("mode") == "relaxed":  # +1
                status = "relaxed"
            else:  # +1
                if config.get("fallback"):  # +1 (nesting=3) +3
                    status = "fallback"
                else:  # +1
                    status = "default"
        elif config.get("disabled"):  # +1
            status = "disabled"
        else:  # +1
            if config.get("auto"):  # +1 (nesting=2) +2
                status = "auto"
            else:  # +1
                status = "manual"
    return status


# --------------------------------------------------------------------------
# 34. simple_pure_h — simple
# --------------------------------------------------------------------------
def simple_pure_h(n: int) -> list:
    return list(range(n))


# --------------------------------------------------------------------------
# 35. complex_with_docstring — COMPLEX [F1]
#     Long docstring (10+ lines).
#     Tests: docstring preservation, start_line past docstring.
# --------------------------------------------------------------------------
def complex_with_docstring(records: list) -> dict:
    """
    Process records and compute aggregate statistics.

    This function takes a list of records where each record is a dict
    with various fields. It computes totals, counts, and categorizations.

    Args:
        records: List of record dicts. Each record should have:
            - "category": str, one of "A", "B", "C"
            - "value": int, the numeric value
            - "tags": list of str, optional tags

    Returns:
        A dict with aggregate stats per category.

    Notes:
        This docstring is intentionally long to test whether
        the fix process preserves it correctly.
    """
    stats: Dict[str, int] = {}
    if records:  # +1
        for rec in records:  # +1 (nesting=1) +1
            cat = rec.get("category", "unknown")
            if cat == "A":  # +1 (nesting=2) +2
                if rec.get("value", 0) > 50:  # +1 (nesting=3) +3
                    stats[cat] = stats.get(cat, 0) + rec["value"]
                    if rec.get("tags"):  # +1 (nesting=4) +4
                        for tag in rec["tags"]:  # +1 (nesting=5) +5
                            if tag.startswith("important"):  # +1 (nesting=6) +6
                                stats["tagged"] = stats.get("tagged", 0) + 1
                elif rec.get("value", 0) > 0:  # +1
                    stats[cat] = stats.get(cat, 0) + 1
            elif cat == "B":  # +1
                if rec.get("value", 0) > 25:  # +1 (nesting=3) +3
                    stats[cat] = stats.get(cat, 0) + rec["value"]
            elif cat == "C":  # +1
                stats[cat] = stats.get(cat, 0) + 1
    return stats


# --------------------------------------------------------------------------
# 36. complex_with_comments — COMPLEX [F2]
#     Inline comments throughout.
#     Tests: comment preservation.
# --------------------------------------------------------------------------
def complex_with_comments(data: list) -> list:
    # Initialize output buffer
    output = []

    if data:  # +1 — check for non-empty input
        for item in data:  # +1 (nesting=1) +1 — iterate all items
            # Handle dict items
            if isinstance(item, dict):  # +1 (nesting=2) +2
                # Check for required key
                if "key" in item:  # +1 (nesting=3) +3
                    # Validate value range
                    if item["key"] > 0:  # +1 (nesting=4) +4
                        # Positive value — add to output
                        output.append(item["key"])
                        # Check for bonus multiplier
                        if item.get("bonus"):  # +1 (nesting=5) +5
                            output.append(item["key"] * item["bonus"])
                    elif item["key"] < 0:  # +1
                        # Negative — skip with warning
                        output.append(0)
                    else:  # +1
                        # Zero — add placeholder
                        output.append(-1)
                else:  # +1
                    # No key found — use fallback
                    output.append(None)
            elif isinstance(item, int):  # +1
                # Raw int — validate range
                if item > 100:  # +1 (nesting=3) +3
                    output.append(item)
                elif item > 0:  # +1
                    output.append(item * 2)
            else:  # +1
                # Unknown type — skip
                pass

    # Return final output
    return output


# ══════════════════════════════════════════════════════════════════════════════
# BETWEEN CLASSES [A6]
# ══════════════════════════════════════════════════════════════════════════════


class EmptyClassA:
    """Empty class before complex_between_classes."""
    pass


# --------------------------------------------------------------------------
# 37. complex_between_classes — COMPLEX [A6]
#     Between two class definitions.
#     Tests: file structure — function amid class defs.
# --------------------------------------------------------------------------
def complex_between_classes(mappings: list) -> dict:
    merged = {}
    if mappings:  # +1
        for m in mappings:  # +1 (nesting=1) +1
            if isinstance(m, dict):  # +1 (nesting=2) +2
                for k, v in m.items():  # +1 (nesting=3) +3
                    if k in merged:  # +1 (nesting=4) +4
                        if isinstance(merged[k], int) and isinstance(v, int):  # +1 (nesting=5) +5
                            merged[k] += v
                        elif isinstance(merged[k], list):  # +1
                            merged[k].append(v)
                        else:  # +1
                            merged[k] = [merged[k], v]
                    else:  # +1
                        merged[k] = v
            elif isinstance(m, list):  # +1
                for item in m:  # +1 (nesting=3) +3
                    if isinstance(item, tuple) and len(item) == 2:  # +1 (nesting=4) +4
                        merged[item[0]] = item[1]
    return merged


class EmptyClassB:
    """Empty class after complex_between_classes."""
    pass


# --------------------------------------------------------------------------
# 38. simple_pure_i — simple
# --------------------------------------------------------------------------
def simple_pure_i(text: str) -> str:
    return text.replace(" ", "_").lower()


# --------------------------------------------------------------------------
# 39. complex_mega — COMPLEX [E6]
#     ~60 lines, complexity 40+.
#     Tests: very large replacement block.
# --------------------------------------------------------------------------
def complex_mega(
    primary: list,
    secondary: Optional[dict] = None,
    tertiary: Optional[list] = None,
    flags: Optional[dict] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "total": 0, "categories": {}, "errors": [], "warnings": [],
    }

    if primary:  # +1
        for idx, item in enumerate(primary):  # +1 (nesting=1) +1
            if isinstance(item, dict):  # +1 (nesting=2) +2
                cat = item.get("category", "default")
                if cat == "critical":  # +1 (nesting=3) +3
                    if item.get("value", 0) > 100:  # +1 (nesting=4) +4
                        report["total"] += item["value"]
                        if item.get("verified"):  # +1 (nesting=5) +5
                            report["categories"].setdefault(cat, []).append(item)
                        else:  # +1
                            report["warnings"].append(f"Unverified critical: {idx}")
                    elif item.get("value", 0) > 0:  # +1
                        report["total"] += 1
                    else:  # +1
                        report["errors"].append(f"Invalid critical value: {idx}")
                elif cat == "normal":  # +1
                    if item.get("value", 0) > 50:  # +1 (nesting=4) +4
                        report["total"] += item["value"]
                    elif item.get("value", 0) > 0:  # +1
                        report["total"] += 1
                elif cat == "low":  # +1
                    report["total"] += 1
                else:  # +1
                    report["warnings"].append(f"Unknown category: {cat}")
            elif isinstance(item, int):  # +1
                if item > 0:  # +1 (nesting=3) +3
                    report["total"] += item
                else:  # +1
                    report["errors"].append(f"Negative int at {idx}")

    if secondary:  # +1
        for key, val in secondary.items():  # +1 (nesting=1) +1
            if isinstance(val, list):  # +1 (nesting=2) +2
                for sub in val:  # +1 (nesting=3) +3
                    if sub > 0:  # +1 (nesting=4) +4
                        report["total"] += sub
                    else:  # +1
                        report["errors"].append(f"Bad secondary: {key}")
            elif isinstance(val, int):  # +1
                if val > 0:  # +1 (nesting=3) +3
                    report["total"] += val

    if tertiary:  # +1
        for t_item in tertiary:  # +1 (nesting=1) +1
            if t_item:  # +1 (nesting=2) +2
                report["total"] += 1

    if flags:  # +1
        if flags.get("double"):  # +1 (nesting=1) +1
            report["total"] *= 2
        if flags.get("cap"):  # +1 (nesting=1) +1
            if report["total"] > flags["cap"]:  # +1 (nesting=2) +2
                report["total"] = flags["cap"]
                report["warnings"].append("Total capped")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# CHAIN DEPENDENCY [B5]
# ══════════════════════════════════════════════════════════════════════════════


# --------------------------------------------------------------------------
# 40. complex_chain_a — COMPLEX [B5]
#     Head of chain: calls #41.
#     Tests: triple dependency chain.
# --------------------------------------------------------------------------
def complex_chain_a(pipeline: list) -> dict:
    result = {"stage_a": 0, "forwarded": 0}
    if pipeline:  # +1
        for step in pipeline:  # +1 (nesting=1) +1
            if isinstance(step, dict):  # +1 (nesting=2) +2
                if step.get("process"):  # +1 (nesting=3) +3
                    if step.get("value", 0) > 10:  # +1 (nesting=4) +4
                        result["stage_a"] += step["value"]
                    else:  # +1
                        result["stage_a"] += 1
                elif step.get("forward"):  # +1
                    sub_result = complex_chain_b(step.get("items", []))
                    if sub_result.get("stage_b", 0) > 0:  # +1 (nesting=4) +4
                        result["forwarded"] += sub_result["stage_b"]
            elif isinstance(step, int):  # +1
                if step > 0:  # +1 (nesting=3) +3
                    result["stage_a"] += step
                else:  # +1
                    result["stage_a"] -= 1
    return result


# --------------------------------------------------------------------------
# 41. complex_chain_b — COMPLEX [B5]
#     Middle of chain: calls #42.
#     Tests: triple dependency chain.
# --------------------------------------------------------------------------
def complex_chain_b(items: list) -> dict:
    result = {"stage_b": 0, "delegated": 0}
    if items:  # +1
        for item in items:  # +1 (nesting=1) +1
            if isinstance(item, dict):  # +1 (nesting=2) +2
                if item.get("handle"):  # +1 (nesting=3) +3
                    if item.get("priority") == "high":  # +1 (nesting=4) +4
                        result["stage_b"] += 10
                    elif item.get("priority") == "medium":  # +1
                        result["stage_b"] += 5
                    else:  # +1
                        result["stage_b"] += 1
                elif item.get("delegate"):  # +1
                    sub = complex_chain_c(item.get("payload", []))
                    if sub.get("stage_c", 0) > 0:  # +1 (nesting=4) +4
                        result["delegated"] += sub["stage_c"]
            elif isinstance(item, int):  # +1
                if item > 0:  # +1 (nesting=3) +3
                    result["stage_b"] += item
    return result


# --------------------------------------------------------------------------
# 42. complex_chain_c — COMPLEX [B5]
#     End of chain (standalone).
#     Tests: end of dependency chain.
# --------------------------------------------------------------------------
def complex_chain_c(payload: list) -> dict:
    result = {"stage_c": 0}
    if payload:  # +1
        for p in payload:  # +1 (nesting=1) +1
            if isinstance(p, dict):  # +1 (nesting=2) +2
                if p.get("active"):  # +1 (nesting=3) +3
                    if p.get("weight", 0) > 5:  # +1 (nesting=4) +4
                        result["stage_c"] += p["weight"]
                    elif p.get("weight", 0) > 0:  # +1
                        result["stage_c"] += 1
                    else:  # +1
                        pass
                elif p.get("fallback"):  # +1
                    result["stage_c"] += 1
            elif isinstance(p, int):  # +1
                if p > 0:  # +1 (nesting=3) +3
                    result["stage_c"] += p
                elif p < 0:  # +1
                    result["stage_c"] -= 1
    return result


# ══════════════════════════════════════════════════════════════════════════════
# EOF [A2]
# ══════════════════════════════════════════════════════════════════════════════


# --------------------------------------------------------------------------
# 43. complex_eof — COMPLEX [A2]
#     Last function in the file. No trailing blank line.
#     Tests: end_line at EOF, replacement handles end-of-file correctly.
# --------------------------------------------------------------------------
def complex_eof(final_data: list) -> Optional[int]:
    total = 0
    if final_data:  # +1
        for item in final_data:  # +1 (nesting=1) +1
            if isinstance(item, dict):  # +1 (nesting=2) +2
                if item.get("count", 0) > 0:  # +1 (nesting=3) +3
                    total += item["count"]
                    if item.get("multiplier"):  # +1 (nesting=4) +4
                        total *= item["multiplier"]
                        if total > 10000:  # +1 (nesting=5) +5
                            return total
                    elif item.get("divisor"):  # +1
                        if item["divisor"] != 0:  # +1 (nesting=5) +5
                            total //= item["divisor"]
                elif item.get("reset"):  # +1
                    total = 0
            elif isinstance(item, int):  # +1
                if item > 0:  # +1 (nesting=3) +3
                    total += item
                elif item < -100:  # +1
                    return None
    if total == 0:  # +1
        return None
    return total