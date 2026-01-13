"""
Evaluation Metrics for KineticAugment.

Provides metrics for evaluating augmentation quality, temporal smoothness,
diversity, and constraint satisfaction.
"""

from kinetic_augment.evaluation.metrics.quality import (
    LimbLengthConsistencyMetric,
    PoseValidityMetric,
    LandmarkDetectionRate,
    AnatomicalPlausibilityMetric,
)

from kinetic_augment.evaluation.metrics.temporal import (
    VelocityMetric,
    SmoothnessMetric,
    TemporalCoherenceMetric,
    SpectralSmoothnessMetric,
)

from kinetic_augment.evaluation.metrics.diversity import (
    VarianceMetric,
    DistributionShiftMetric,
    AugmentationDiversityMetric,
    CoverageMetric,
)

from kinetic_augment.evaluation.metrics.constraint import (
    ConstraintSatisfactionMetric,
    ViolationSeverityMetric,
    JointLimitComplianceMetric,
)

__all__ = [
    # Quality
    'LimbLengthConsistencyMetric',
    'PoseValidityMetric',
    'LandmarkDetectionRate',
    'AnatomicalPlausibilityMetric',
    # Temporal
    'VelocityMetric',
    'SmoothnessMetric',
    'TemporalCoherenceMetric',
    'SpectralSmoothnessMetric',
    # Diversity
    'VarianceMetric',
    'DistributionShiftMetric',
    'AugmentationDiversityMetric',
    'CoverageMetric',
    # Constraint
    'ConstraintSatisfactionMetric',
    'ViolationSeverityMetric',
    'JointLimitComplianceMetric',
]
