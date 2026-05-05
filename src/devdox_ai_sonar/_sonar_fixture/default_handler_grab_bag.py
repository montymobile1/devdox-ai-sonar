"""Fixture for the DefaultRuleHandler catch-all path.

Bundles four rules that have no specialized handler, so all findings
flow through DefaultRuleHandler (rule_handler.py:1479):
- python:S1481 (unused local variable)
- python:S125  (commented-out code)
- python:S1186 (empty function body)
- python:S1135 (TODO comment)

Routing: DefaultRuleHandler -> DEFAULT agent prompts.
"""


def compute_average(values):
    count = len(values)
    total = sum(values)
    if not values:
        return 0
    return total / len(values)


def render_user(user):
    # if user.is_admin:
    #     return f"<admin>{user.name}</admin>"
    # else:
    #     return f"<user>{user.name}</user>"
    return user.name


def placeholder():
    pass


def settle_invoice(invoice):
    # TODO: handle multi-currency totals
    return invoice.total
