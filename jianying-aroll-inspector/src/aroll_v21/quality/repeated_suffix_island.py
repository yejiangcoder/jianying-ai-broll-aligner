from __future__ import annotations


COORDINATION_CONNECTORS = {"和", "与", "及", "以及", "或", "或者", "跟", "同"}


def is_coordinated_parallel_suffix_repetition(tokens: list[str], left_start: int, suffix_start: int, size: int) -> bool:
    if size <= 0 or left_start < 0 or suffix_start <= left_start:
        return False
    if tokens[left_start : left_start + size] != tokens[suffix_start : suffix_start + size]:
        return False
    between = [token for token in tokens[left_start + size : suffix_start] if token]
    if not between or not any(token in COORDINATION_CONNECTORS for token in between):
        return False
    return any(token not in COORDINATION_CONNECTORS for token in between)
