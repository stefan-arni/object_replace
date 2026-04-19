"""Time/token-dependent attention swap schedules. The novel contribution.

A `Schedule` maps `t_frac in [0, 1]` (sampling progress, 0 = first step at high
noise, 1 = last step at clean) to a swap weight in [0, 1]. The weight controls
how much of source's cross-attention to blend into target's at that timestep:
    swap_weight = 1.0  -> use source attention (preserve identity)
    swap_weight = 0.0  -> use target attention (let the swap happen)

A `ScheduleSet` bundles three schedules, one per token role:
    preserved : tokens that match between source and target prompts
    replaced  : tokens that differ -- the swap (cat <-> dog)
    context   : special tokens (<sos>, <eos>, <pad>) -- weak attention anyway

Vanilla P2P is a special case: `Step(tau, 1.0, 0.0)` for preserved, `Constant(0.0)`
for replaced. The hypothesis under test is that *non*-step shapes for the
'replaced' role beat the vanilla baseline, and that the right shape depends on
edit type (structural vs geometric).
"""
import math
from dataclasses import dataclass, field


class Schedule:
    def __call__(self, t_frac: float) -> float:
        raise NotImplementedError


class Constant(Schedule):
    def __init__(self, value: float):
        self.value = value

    def __call__(self, t_frac):
        return self.value


class LinearDecay(Schedule):
    def __init__(self, start: float = 1.0, end: float = 0.0):
        self.start = start
        self.end = end

    def __call__(self, t_frac):
        return self.start + (self.end - self.start) * t_frac


class Cosine(Schedule):
    """Smooth half-cosine from `start` (at t_frac=0) to `end` (at t_frac=1)."""

    def __init__(self, start: float = 1.0, end: float = 0.0):
        self.start = start
        self.end = end

    def __call__(self, t_frac):
        c = (1 + math.cos(math.pi * t_frac)) / 2  # 1 at 0, 0 at 1
        return self.end + (self.start - self.end) * c


class Step(Schedule):
    """Vanilla-P2P shape: `before` while t_frac < threshold, `after` otherwise."""

    def __init__(self, threshold: float, before: float = 1.0, after: float = 0.0):
        self.threshold = threshold
        self.before = before
        self.after = after

    def __call__(self, t_frac):
        return self.before if t_frac < self.threshold else self.after


class Piecewise(Schedule):
    """Piecewise-linear interpolation through (t_frac, weight) control points.

    The 'learned' shape from the spec: sweep 3-4 inflection points to fit
    the optimum for a given edit type.
    """

    def __init__(self, points: list[tuple[float, float]]):
        pts = sorted(points)
        if pts[0][0] > 0.0:
            pts = [(0.0, pts[0][1])] + pts
        if pts[-1][0] < 1.0:
            pts = pts + [(1.0, pts[-1][1])]
        self.points = pts

    def __call__(self, t_frac):
        for i in range(len(self.points) - 1):
            t0, w0 = self.points[i]
            t1, w1 = self.points[i + 1]
            if t0 <= t_frac <= t1:
                if t1 == t0:
                    return w0
                return w0 + (w1 - w0) * (t_frac - t0) / (t1 - t0)
        return self.points[-1][1]


@dataclass
class ScheduleSet:
    preserved: Schedule
    replaced: Schedule
    context: Schedule = field(default_factory=lambda: Constant(1.0))

    def __call__(self, t_frac: float, role: str) -> float:
        if role == "preserved":
            return self.preserved(t_frac)
        if role == "replaced":
            return self.replaced(t_frac)
        if role == "context":
            return self.context(t_frac)
        raise ValueError(f"unknown token role: {role!r}")


# ---- presets -----------------------------------------------------------------

def vanilla_p2p(tau: float = 0.8) -> ScheduleSet:
    """Reference baseline. Identical to P2PReplaceController(tau=tau)."""
    return ScheduleSet(
        preserved=Step(tau, before=1.0, after=0.0),
        replaced=Constant(0.0),
        context=Step(tau, before=1.0, after=0.0),
    )


def linear_decay_replaced() -> ScheduleSet:
    """Hypothesis: structural edits (cat <-> dog) want fast decay on replaced --
    target token grabs texture early while source attention still drives shape."""
    return ScheduleSet(
        preserved=Constant(1.0),
        replaced=LinearDecay(start=1.0, end=0.0),
    )


def cosine_replaced() -> ScheduleSet:
    return ScheduleSet(
        preserved=Constant(1.0),
        replaced=Cosine(start=1.0, end=0.0),
    )


def constant_replaced(level: float = 0.5) -> ScheduleSet:
    """Hypothesis: geometric edits (apple <-> banana) want layout to adapt
    gradually -- a steady mid-level swap on replaced lets shape rearrange."""
    return ScheduleSet(
        preserved=Constant(1.0),
        replaced=Constant(level),
    )


def piecewise_demo() -> ScheduleSet:
    """Three-inflection-point example for the learned-schedule sweep."""
    return ScheduleSet(
        preserved=Constant(1.0),
        replaced=Piecewise([(0.0, 1.0), (0.3, 0.7), (0.7, 0.2), (1.0, 0.0)]),
    )
