"""Fewest-forward windows for complexes longer than the inference budget.

Cutting both chains multiplies the number of model calls, so the planner
searches increasing round counts and keeps a chain whole whenever that
fits. At the same round count, a one-chain split is preferred over a
two-chain split: one chain into 4 is tried before both chains into 2.
"""

from __future__ import annotations


def _ceil_div(numerator: int, denominator: int) -> int:
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def contiguous_spans(length: int, count: int) -> list[tuple[int, int]]:
    """Split ``[0, length)`` into ``count`` non-overlapping spans.

    Span lengths differ by at most one, and the longer spans come first.
    """
    length = int(length)
    count = int(count)
    if length <= 0:
        raise ValueError(f"sequence length must be positive, got {length}.")
    if count <= 0:
        raise ValueError(f"window count must be positive, got {count}.")
    if count > length:
        raise ValueError(
            f"Cannot split length {length} into {count} non-empty windows."
        )
    base, extra = divmod(length, count)
    spans: list[tuple[int, int]] = []
    start = 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        spans.append((start, start + size))
        start += size
    return spans


def choose_window_counts(length_1: int, length_2: int, budget: int) -> tuple[int, int]:
    """Return how many windows each chain needs to minimize the forward count.

    ``k1 * k2`` is the number of forwards. A split is feasible when
    ``ceil(length_1 / k1) + ceil(length_2 / k2) <= budget``. Ties prefer
    cutting only one chain, and then cutting the longer chain.
    """
    length_1 = int(length_1)
    length_2 = int(length_2)
    budget = int(budget)
    if length_1 <= 0 or length_2 <= 0:
        raise ValueError(
            f"sequence lengths must be positive, got {length_1} and {length_2}."
        )
    if length_1 + length_2 <= budget:
        return 1, 1
    if budget < 2:
        raise ValueError(
            f"max_full_length must be at least 2 to tile a complex of "
            f"{length_1}+{length_2}, got {budget}."
        )

    best: tuple[int, int, int, int, int] | None = None
    for window_1 in range(1, min(length_1, budget - 1) + 1):
        window_2 = min(length_2, budget - window_1)
        if window_2 < 1:
            continue
        count_1 = _ceil_div(length_1, window_1)
        count_2 = _ceil_div(length_2, window_2)
        rounds = count_1 * count_2
        single_chain_rank = min(count_1, count_2)
        if length_1 == length_2 or (length_1 > length_2 and count_1 >= count_2):
            longer_penalty = 0
        elif length_2 > length_1 and count_2 >= count_1:
            longer_penalty = 0
        else:
            longer_penalty = 1
        candidate = (rounds, single_chain_rank, longer_penalty, count_1, count_2)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError(
            f"No window split fits lengths {length_1}+{length_2} "
            f"within budget {budget}."
        )
    return best[3], best[4]


def plan_ppi_windows(
    length_1: int,
    length_2: int,
    budget: int | None,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Window pairs ``((start_1, end_1), (start_2, end_2))`` covering both chains.

    ``budget`` is the maximum concatenated length of one forward. ``None``
    keeps the complex in a single window. Every returned pair sums to at
    most ``budget`` residues, and every inter-chain residue pair falls in
    exactly one window.
    """
    length_1 = int(length_1)
    length_2 = int(length_2)
    if length_1 <= 0 or length_2 <= 0:
        raise ValueError(
            f"sequence lengths must be positive, got {length_1} and {length_2}."
        )
    if budget is None or length_1 + length_2 <= int(budget):
        return [((0, length_1), (0, length_2))]

    budget = int(budget)
    count_1, count_2 = choose_window_counts(length_1, length_2, budget)
    spans_1 = contiguous_spans(length_1, count_1)
    spans_2 = contiguous_spans(length_2, count_2)
    windows = [(span_1, span_2) for span_1 in spans_1 for span_2 in spans_2]
    for (start_1, end_1), (start_2, end_2) in windows:
        window_length = (end_1 - start_1) + (end_2 - start_2)
        if window_length > budget:
            raise RuntimeError(
                f"Planned window length {window_length} exceeds budget {budget}."
            )
    return windows
