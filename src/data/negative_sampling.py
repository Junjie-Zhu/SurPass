"""Exact indexed matching and read-only multiprocessing for hard negatives."""

from __future__ import annotations

import hashlib
import heapq
import signal
from bisect import bisect_left
from dataclasses import dataclass


def stable_tie_breaker(seed: int, *values) -> str:
    payload = "|".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NegativeQuery:
    anchor: str
    target: str
    source: str
    split: str
    pair_id: str | None
    reverse: bool
    blocked: frozenset[str]
    blocked_families: frozenset[str] = frozenset()


# Each degree has sorted length buckets. Candidates in a bucket share exactly
# the same score; only eligible candidates at the best score need tie hashing.
Pool = tuple[tuple[int, tuple[int, ...], tuple[list[str], ...]], ...]
Proposal = tuple[float, str] | None


@dataclass
class CandidateIndex:
    families: dict[str, str]
    lengths: dict[str, int]
    degrees: dict[str, int]
    family_neighbors: dict[str, set[str]]
    existing_neighbors: dict[str, set[str]]
    pools: dict[tuple[str, str | None], Pool]
    seed: int

    def best(self, query: NegativeQuery) -> Proposal:
        anchor_family = self.families[query.anchor]
        target_length = self.lengths[query.target]
        target_degree = self.degrees[self.families[query.target]]
        original_neighbors = self.existing_neighbors.get(query.anchor, ())
        positive_neighbors = self.family_neighbors.get(anchor_family, ())
        pools = (
            self.pools.get((query.source, query.split), ()),
            # Preserve the legacy fallback for families without a split.
            self.pools.get((query.source, None), ()),
        )
        heap = []

        def push(pool_index, group_index, index, direction):
            degree, lengths, _ = pools[pool_index][group_index]
            if 0 <= index < len(lengths):
                score = abs(lengths[index] - target_length) / max(target_length, 1)
                score += abs(degree - target_degree) / max(target_degree, 1)
                heapq.heappush(heap, (score, pool_index, group_index, index, direction))

        for pool_index, pool in enumerate(pools):
            for group_index, (_, lengths, _) in enumerate(pool):
                pivot = bisect_left(lengths, target_length)
                push(pool_index, group_index, pivot, 1)
                push(pool_index, group_index, pivot - 1, -1)

        best = None
        while heap:
            score, pool_index, group_index, index, direction = heapq.heappop(heap)
            if best is not None and score > best[0]:
                break
            _, _, buckets = pools[pool_index][group_index]
            for candidate in buckets[index]:
                family = self.families[candidate]
                if candidate == query.anchor or family == anchor_family:
                    continue
                if family in positive_neighbors or family in query.blocked_families:
                    continue
                if candidate in original_neighbors or candidate in query.blocked:
                    continue
                tie = stable_tie_breaker(self.seed, query.pair_id, query.reverse, candidate)
                ranked = (score, tie, candidate)
                if best is None or ranked < best:
                    best = ranked
            push(pool_index, group_index, index + direction, direction)
        return None if best is None else (best[0], best[2])


_WORKER_INDEX: CandidateIndex | None = None


def init_candidate_worker(index: CandidateIndex) -> None:
    global _WORKER_INDEX
    # The parent owns cancellation and terminates the entire pool cleanly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER_INDEX = index


def rank_candidate_queries(queries: tuple[NegativeQuery | None, ...]) -> tuple[Proposal, ...]:
    """Rank the preferred orientation first; avoid unused reverse searches.

    If its proposal later conflicts with an earlier parent commit and no
    replacement exists, the parent searches the reverse orientation itself.
    """
    if _WORKER_INDEX is None:
        raise RuntimeError("Negative candidate worker has not been initialized")
    first, second = queries
    preferred = None if first is None else _WORKER_INDEX.best(first)
    reverse = None if preferred is not None or second is None else _WORKER_INDEX.best(second)
    return preferred, reverse
