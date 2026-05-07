"""Fixture for the ConvenationNameHandler path.

Targets the three rules dispatched to ConvenationNameHandler
(rule_handler.py:237):
- python:S117  (variable / parameter naming - default regex ^[_a-z][_a-z0-9]*$)
- python:S1172 (unused parameter)
- python:S1542 (function name should comply with naming convention)

Routing: ConvenationNameHandler -> DEFAULT agent prompts.
"""


def compute_total(BasePrice, TaxRate):
    Total = BasePrice * (1 + TaxRate)
    return Total


def format_label(UserName, MaxLen):
    Truncated = UserName[:MaxLen]
    return Truncated


def grade_score(RawScore, MultiplierUnused):
    AdjustedScore = RawScore + 5
    return AdjustedScore


def ProcessRecord(record):
    return record
