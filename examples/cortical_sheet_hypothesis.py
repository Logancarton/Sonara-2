from __future__ import annotations

import time

import numpy as np

from experiments.cortical_sheet import FastCorticalSheet, StreamSpec
from experiments.cortical_sheet_benchmark import (
    completion_trial,
    normalize,
    separation_trial,
)


def main() -> None:
    print("SONARA FAST CORTICAL-SHEET HYPOTHESIS PROBE")
    print("=" * 52)

    stream_sets = (
        ("sensory",),
        ("sensory", "context"),
        ("sensory", "context", "body"),
        ("sensory", "context", "body", "time"),
    )
    print("\nAmbiguous-state separation as independent streams converge:")
    for present in stream_sets:
        values = [separation_trial(seed, present) for seed in range(6)]
        print(f"  {len(present)} stream(s): separation={np.mean(values):.3f}")

    recurrent = np.asarray([completion_trial(seed, True) for seed in range(6)])
    no_recurrence = np.asarray([completion_trial(seed, False) for seed in range(6)])
    print("\nPartial-cue completion diagnostic (sensory + context only):")
    print(
        "  recurrence ON : "
        f"accuracy={np.mean(recurrent[:, 0]):.3f}, "
        f"margin={np.mean(recurrent[:, 1]):.3f}"
    )
    print(
        "  recurrence OFF: "
        f"accuracy={np.mean(no_recurrence[:, 0]):.3f}, "
        f"margin={np.mean(no_recurrence[:, 1]):.3f}"
    )
    print("  completion gate: NOT YET PROVED")

    feature_size = 16
    streams = (
        StreamSpec("visual", feature_size, (0.05, 0.85), gain=4.0),
        StreamSpec("auditory", feature_size, (0.95, 0.75)),
        StreamSpec("body", feature_size, (0.05, 0.15)),
        StreamSpec("context", feature_size, (0.95, 0.15)),
    )
    sheet = FastCorticalSheet(
        100,
        100,
        streams,
        sparsity=0.02,
        seed=21,
    )
    rng = np.random.default_rng(21)
    inputs = {
        stream.name: normalize(rng.normal(size=feature_size))
        for stream in streams
    }
    started = time.perf_counter()
    for tick in range(100):
        sheet.step(inputs if tick < 20 else {}, learn=tick < 20)
    elapsed = time.perf_counter() - started
    print("\nFast-math scale probe:")
    print(f"  units={sheet.size:,}")
    print(f"  recurrent edges={sheet.recurrent_edge_count:,}")
    print(f"  winner budget={sheet.winner_budget:,} ({sheet.sparsity:.1%})")
    print(f"  100 update steps={elapsed:.3f}s")


if __name__ == "__main__":
    main()
