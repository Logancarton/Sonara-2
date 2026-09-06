from __future__ import annotations

from typing import Mapping

import numpy as np

from .cortical_sheet import FastCorticalSheet, SheetStep, SignalBundle


class FeedbackCorticalSheet(FastCorticalSheet):
    """
    Experimental cortical sheet that accepts a delayed population return bundle.

    This keeps cortical state, competition, recurrent propagation, and learning
    in the cortical owner. The memory branch can only contribute current; it
    cannot directly mutate cortical state or select cortical winners.
    """

    def step(
        self,
        inputs: Mapping[str, np.ndarray] | None = None,
        *,
        feedback_bundle: SignalBundle | None = None,
        feedback_gain: float = 1.0,
        dt_ms: float = 1.0,
        learn: bool = False,
        recurrent: bool = True,
    ) -> SheetStep:
        if dt_ms <= 0.0:
            raise ValueError("dt_ms must be > 0")
        if feedback_gain < 0.0:
            raise ValueError("feedback_gain must be >= 0")
        self.time_ms += float(dt_ms)

        direct_drive, branches, normalized_inputs = self.convergent_drive(inputs)
        recurrent_drive = self.recurrent_current() if recurrent else 0.0
        if feedback_bundle is None:
            feedback_drive: np.ndarray | float = 0.0
        else:
            feedback_drive = (
                feedback_bundle.as_dense(self.size) * float(feedback_gain)
            )
        target_state = direct_drive + recurrent_drive + feedback_drive

        alpha = 1.0 - np.exp(-float(dt_ms) / np.maximum(self.tau_ms, 1e-6))
        self.state += alpha.astype(np.float32) * (target_state - self.state)

        competition_score = np.maximum(self.state - self.firing_threshold, 0.0)
        competition_score /= 1.0 + self.homeostatic_pressure * self.usage
        positive = np.flatnonzero(competition_score > 0.0)
        if positive.size > self.winner_budget:
            selected = positive[
                np.argpartition(
                    competition_score[positive],
                    -self.winner_budget,
                )[-self.winner_budget :]
            ]
            selected = selected[
                np.argsort(-competition_score[selected], kind="stable")
            ]
        else:
            selected = positive[
                np.argsort(-competition_score[positive], kind="stable")
            ]

        self.activity.fill(0.0)
        if selected.size:
            selected_scores = competition_score[selected]
            peak = max(float(np.max(selected_scores)), 1e-12)
            self.activity[selected] = selected_scores / peak

        self.usage = (
            0.995 * self.usage + 0.005 * (self.activity > 0.0).astype(np.float32)
        )

        if learn and selected.size:
            self._learn_afferents(selected, branches, normalized_inputs)
            self._learn_recurrent_edges()

        bundle = self.current_bundle()
        return SheetStep(
            time_ms=self.time_ms,
            winner_indices=selected.copy(),
            winner_activity=self.activity[selected].copy(),
            active_fraction=float(selected.size / self.size),
            mean_winner_coldness=(
                float(np.mean(self.coldness[selected])) if selected.size else 0.0
            ),
            bundle=bundle,
        )
