import numpy as np

from sonara.neural import SparseRepresentationField


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _overlap(left: np.ndarray, right: np.ndarray, winner_count: int) -> float:
    return float(np.intersect1d(left, right).size / winner_count)


def _correlated_prototypes(
    rng: np.random.Generator,
    families: int,
    feature_size: int,
    common_strength: float,
) -> np.ndarray:
    common = _normalize(rng.normal(size=feature_size))
    unique = []
    for _ in range(families):
        vector = rng.normal(size=feature_size)
        vector -= common * np.dot(vector, common)
        unique.append(_normalize(vector))

    unique_strength = float(np.sqrt(1.0 - common_strength**2))
    return np.asarray(
        [_normalize(common_strength * common + unique_strength * vector) for vector in unique]
    )


def _trial(seed: int, common_strength: float) -> tuple[float, float, float, float]:
    families = 12
    feature_size = 24
    population_size = 384
    winner_count = 24
    training_examples = 24
    evaluation_examples = 8
    training_noise = 0.34
    evaluation_noise = 0.42

    rng = np.random.default_rng(seed)
    prototypes = _correlated_prototypes(
        rng,
        families=families,
        feature_size=feature_size,
        common_strength=common_strength,
    )

    frozen = SparseRepresentationField(
        "frozen", "R", population_size, feature_size,
        winner_count=winner_count, seed=1000 + seed, learning_rate=0.08,
    )
    learned = SparseRepresentationField(
        "learned", "R", population_size, feature_size,
        winner_count=winner_count, seed=1000 + seed, learning_rate=0.08,
    )

    for family in range(families):
        for _ in range(training_examples):
            sample = _normalize(
                prototypes[family]
                + training_noise * rng.normal(size=feature_size) / np.sqrt(feature_size)
            )
            frozen.encode(sample, learn=False)
            learned.encode(sample, learn=True)

    evaluation_sets = [
        [
            _normalize(
                prototype
                + evaluation_noise * rng.normal(size=feature_size) / np.sqrt(feature_size)
            )
            for _ in range(evaluation_examples)
        ]
        for prototype in prototypes
    ]

    def separation(field: SparseRepresentationField) -> tuple[float, float]:
        activations = [
            [field.encode(sample, learn=False).winner_indices for sample in samples]
            for samples in evaluation_sets
        ]
        within = []
        between = []

        for family in range(families):
            for left in range(evaluation_examples):
                for right in range(left + 1, evaluation_examples):
                    within.append(
                        _overlap(
                            activations[family][left],
                            activations[family][right],
                            winner_count,
                        )
                    )

        for left_family in range(families):
            for right_family in range(left_family + 1, families):
                for example in range(evaluation_examples):
                    between.append(
                        _overlap(
                            activations[left_family][example],
                            activations[right_family][example],
                            winner_count,
                        )
                    )

        return float(np.mean(within)), float(np.mean(between))

    frozen_within, frozen_between = separation(frozen)
    learned_within, learned_between = separation(learned)
    return frozen_within, frozen_between, learned_within, learned_between


def _aggregate(common_strength: float, seeds: range = range(8)) -> dict[str, float]:
    rows = np.asarray([_trial(seed, common_strength) for seed in seeds])
    frozen_within = float(np.mean(rows[:, 0]))
    frozen_between = float(np.mean(rows[:, 1]))
    learned_within = float(np.mean(rows[:, 2]))
    learned_between = float(np.mean(rows[:, 3]))
    return {
        "frozen_within": frozen_within,
        "frozen_between": frozen_between,
        "frozen_separation": frozen_within - frozen_between,
        "learned_within": learned_within,
        "learned_between": learned_between,
        "learned_separation": learned_within - learned_between,
    }


def test_learning_improves_separation_when_families_are_well_separated():
    metrics = _aggregate(common_strength=0.40)
    assert metrics["learned_separation"] > metrics["frozen_separation"]


def test_learning_must_not_collapse_moderately_correlated_families():
    metrics = _aggregate(common_strength=0.72)
    assert metrics["learned_separation"] > metrics["frozen_separation"], metrics
