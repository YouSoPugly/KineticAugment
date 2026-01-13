"""
Benchmark Module for KineticAugment Evaluation.

Provides tools for running augmentation experiments and generating reports.
"""

from kinetic_augment.evaluation.benchmark.runner import (
    ExperimentConfig,
    ExperimentResult,
    BenchmarkRunner,
)

from kinetic_augment.evaluation.benchmark.experiments import (
    PRESET_COMPARISON,
    INDIVIDUAL_AUGMENTATIONS,
    CONSTRAINT_ABLATION,
    QUICK_TEST,
    create_augmentation_factor_experiments,
    get_standard_experiments,
)

from kinetic_augment.evaluation.benchmark.reports import BenchmarkReport

__all__ = [
    # Runner
    'ExperimentConfig',
    'ExperimentResult',
    'BenchmarkRunner',
    # Predefined experiments
    'PRESET_COMPARISON',
    'INDIVIDUAL_AUGMENTATIONS',
    'CONSTRAINT_ABLATION',
    'QUICK_TEST',
    'create_augmentation_factor_experiments',
    'get_standard_experiments',
    # Reports
    'BenchmarkReport',
]
