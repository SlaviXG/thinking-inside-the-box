from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# AML pattern detection on locally-observable edges.
#
# This module is deliberately bank-scoped: it consumes only the edges a single
# federation node can see (full outgoing from its own accounts, intra-bank
# incoming, intra-bank 2-hop chains). Cross-bank chains are outside the
# node's privacy boundary by design - the architecture trades cross-bank
# visibility for raw-data confidentiality, and local structural patterns
# are the signal we recover without crossing that boundary.


@dataclass
class OutEdge:
    to_id: str
    to_bank: int
    amount: float
    timestamp: Optional[datetime]


@dataclass
class InEdge:
    from_id: str
    amount: float
    timestamp: Optional[datetime]


@dataclass
class Chain:
    mid: str
    dest: str
    amt1: float
    amt2: float
    timestamp: Optional[datetime]


@dataclass
class Pattern:
    name: str
    evidence: str


_TS_FORMATS = (
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
)


def parse_timestamp(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    s = str(ts).strip()
    if not s:
        return None
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _pass_through(outgoing: list[OutEdge], incoming: list[InEdge]) -> list[Pattern]:
    # Incoming amount X followed by outgoing within 48h and within 10% of X.
    # Classic hop-through: funds arrive and immediately move on.
    patterns = []
    used_out = set()
    for inc in incoming:
        if inc.timestamp is None or inc.amount <= 0:
            continue
        for idx, out in enumerate(outgoing):
            if idx in used_out or out.timestamp is None:
                continue
            dt = (out.timestamp - inc.timestamp).total_seconds()
            if dt <= 0 or dt > 48 * 3600:
                continue
            if abs(out.amount - inc.amount) / inc.amount < 0.10:
                patterns.append(Pattern(
                    "pass-through",
                    f"{inc.amount:,.0f} in from {inc.from_id} then {out.amount:,.0f} "
                    f"out to {out.to_id} within {dt / 3600:.1f}h"
                ))
                used_out.add(idx)
                break
    return patterns


def _structuring(outgoing: list[OutEdge]) -> list[Pattern]:
    # >=3 similar-sized outgoing tx to the same destination, each under 10k.
    # Smurfing signature: splitting a transfer to stay below reporting thresholds.
    by_dest: dict[str, list[float]] = {}
    for out in outgoing:
        by_dest.setdefault(out.to_id, []).append(out.amount)
    patterns = []
    for dest, amts in by_dest.items():
        if len(amts) < 3:
            continue
        mean = sum(amts) / len(amts)
        if mean <= 0 or mean >= 10_000:
            continue
        var = sum((a - mean) ** 2 for a in amts) / len(amts)
        stddev = var ** 0.5
        if stddev / mean < 0.20:
            patterns.append(Pattern(
                "structuring",
                f"{len(amts)} similar tx (~{mean:,.0f} each, under 10k) to {dest}"
            ))
    return patterns


def _fan_out(outgoing: list[OutEdge]) -> list[Pattern]:
    # 5+ distinct destinations - diversified outflow often indicates layering.
    unique_dest = {out.to_id for out in outgoing}
    if len(unique_dest) >= 5:
        total = sum(out.amount for out in outgoing)
        return [Pattern(
            "high-fan-out",
            f"{len(unique_dest)} distinct destinations across {len(outgoing)} "
            f"outgoing tx totalling {total:,.0f}"
        )]
    return []


def _rapid_burst(outgoing: list[OutEdge]) -> list[Pattern]:
    # 3+ outgoing tx within a 1h window - velocity signal.
    timed = sorted(
        [o for o in outgoing if o.timestamp is not None],
        key=lambda o: o.timestamp,
    )
    for i in range(len(timed) - 2):
        window = timed[i:i + 3]
        span = (window[-1].timestamp - window[0].timestamp).total_seconds()
        if span <= 3600:
            total = sum(w.amount for w in window)
            return [Pattern(
                "rapid-burst",
                f"3 outgoing tx totalling {total:,.0f} within "
                f"{span / 60:.0f}min starting {window[0].timestamp:%Y-%m-%d %H:%M}"
            )]
    return []


def _cycle(
    account_id: str,
    incoming: list[InEdge],
    chains: list[Chain],
) -> list[Pattern]:
    # Intra-bank cycle: our 2-hop chain terminates at a sender we received from.
    # A -> B -> C where C later (or previously) sent funds back to A.
    sender_ids = {inc.from_id for inc in incoming}
    for ch in chains:
        if ch.dest in sender_ids and ch.dest != account_id:
            return [Pattern(
                "cycle",
                f"{account_id} -> {ch.mid} -> {ch.dest}, and {ch.dest} "
                f"also appears as a sender to {account_id} (intra-bank)"
            )]
    return []


def detect_patterns(
    account_id: str,
    outgoing: list[OutEdge],
    incoming: list[InEdge],
    chains: list[Chain],
) -> list[Pattern]:
    """
    Run all locally-observable pattern detectors on a single account's edges.
    Returns a de-duplicated list ordered by detection priority.
    """
    patterns = []
    patterns.extend(_pass_through(outgoing, incoming))
    patterns.extend(_structuring(outgoing))
    patterns.extend(_cycle(account_id, incoming, chains))
    patterns.extend(_rapid_burst(outgoing))
    patterns.extend(_fan_out(outgoing))
    return patterns


def format_patterns(
    account_id: str,
    bank_id: int,
    patterns: list[Pattern],
    cross_bank_note: Optional[str] = None,
) -> str:
    """
    Render detected patterns as labelled lines for prompt injection.
    Includes the bank-scoped framing so the model (and later readers) see
    that the absence of patterns is a local observation, not a global one.
    """
    lines = [f"Account {account_id} (Bank {bank_id}) - locally-detected AML patterns:"]
    if patterns:
        for p in patterns:
            lines.append(f"- DETECTED {p.name}: {p.evidence}")
    else:
        lines.append("- No structural AML patterns detected within bank scope.")
    if cross_bank_note:
        lines.append(cross_bank_note)
    return "\n".join(lines)
