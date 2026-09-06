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

    The live feedback experiment also carries a short recurrent eligibility
    trace. Recently active cortical sources can therefore strengthen an existing
    route when its downstream target joins the cascade a few milliseconds later.
    """

    def __init__(
        self,
        *args,
        recurrent_eligibility_decay: float = 0.82,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= recurrent_eligibility_decay < 1.0:
            raise ValueError("recurrent_eligibility_decay must be in [0, 1)")
        self.recurrent_eligibility_decay = float(recurrent_eligibility_decay)
        self.recurrent_eligibility = np.zeros(self.size, dtype=np.float32)

    def _update_recurrent_eligibility(self) -> None:
        self.recurrent_eligibility *= self.recurrent_eligibility_decay
        np.maximum(
            self.recurrent_eligibility,
            self.activity,
            out=self.recurrent_eligibility,
        )

    def _learn_temporal_recurrent_edges(self) -> None:
        post = self.activity[self.target_edges]
        pre = self.recurrent_eligibility[self.source_edges]
        eligible = (pre > 0.0) & (post > 0.0)
        if not np.any(eligible):
            return
        coactivity = pre[eligible] * post[eligible]
        self.recurrent_weights[eligible] += (
            self.recurrent_learning_rate
            * coactivity
            * (1.0 - self.recurrent_weights[eligible])
        )
        np.clip(self.recurrent_weights, 0.0, 1.0, out=self.recurrent_weights)

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

        self._update_recurrent_eligibility()

        self.usage = (
            0.995 * self.usage + 0.005 * (self.activity > 0.0).astype(np.float32)
        )

        if learn and selected.size:
            self._learn_afferents(selected, branches, normalized_inputs)
            self._learn_temporal_recurrent_edges()

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

    def reset_state(self, *, reset_usage: bool = False) -> None:
        super().reset_state(reset_usage=reset_usage)
        self.recurrent_eligibility.fill(0.0)
