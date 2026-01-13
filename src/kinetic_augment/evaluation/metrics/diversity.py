"""
Diversity Metrics for KineticAugment Evaluation.

Metrics for evaluating augmentation diversity and distribution characteristics.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from scipy import stats

from kinetic_augment.evaluation.base import BaseMetric, ComparisonMetric, MetricResult


class VarianceMetric(BaseMetric):
    """
    Measure variance/spread of pose data.

    Computes variance statistics to understand the spread
    of augmented samples.
    """

    name = "VarianceMetric"
    description = "Measures data variance and spread"

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """
        Compute variance metrics.

        Args:
            data: Landmarks of shape (frames, landmarks, 3) or
                  batch of shape (batch, frames, landmarks, 3)

        Returns:
            MetricResult with variance statistics
        """
        if data.ndim < 3:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': f'Expected at least 3D array, got {data.ndim}D'}
            )

        # Flatten to (samples, features) for analysis
        if data.ndim == 3:
            flat_data = data.reshape(-1, data.shape[-1])
        else:
            flat_data = data.reshape(-1, data.shape[-1])

        # Overall variance
        total_variance = np.var(flat_data)

        # Per-coordinate variance
        per_coord_variance = np.var(flat_data, axis=0)

        # Effective dimensionality via PCA
        # Percentage of variance explained by top components
        centered = flat_data - np.mean(flat_data, axis=0)
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            explained_variance = (s ** 2) / np.sum(s ** 2)
            cumsum = np.cumsum(explained_variance)
            effective_dims = np.searchsorted(cumsum, 0.95) + 1
        except:
            effective_dims = flat_data.shape[1]

        return MetricResult(
            name=self.name,
            value=float(total_variance),
            details={
                'total_variance': float(total_variance),
                'mean_per_coord_variance': float(np.mean(per_coord_variance)),
                'max_per_coord_variance': float(np.max(per_coord_variance)),
                'effective_dimensions': int(effective_dims),
                'total_dimensions': int(flat_data.shape[1]),
            }
        )


class DistributionShiftMetric(ComparisonMetric):
    """
    Measure distribution shift between original and augmented data.

    Computes statistical distances to quantify how much augmentation
    changes the data distribution.
    """

    name = "DistributionShift"
    description = "Measures distribution shift from augmentation"

    def compute_comparison(
        self,
        original: np.ndarray,
        augmented: np.ndarray,
        **kwargs,
    ) -> MetricResult:
        """
        Compute distribution shift metrics.

        Args:
            original: Original data
            augmented: Augmented data

        Returns:
            MetricResult with distribution shift measures
        """
        # Flatten both to 2D for comparison
        orig_flat = original.reshape(-1, original.shape[-1])
        aug_flat = augmented.reshape(-1, augmented.shape[-1])

        # Sample if too large
        max_samples = 10000
        if len(orig_flat) > max_samples:
            idx = np.random.choice(len(orig_flat), max_samples, replace=False)
            orig_flat = orig_flat[idx]
        if len(aug_flat) > max_samples:
            idx = np.random.choice(len(aug_flat), max_samples, replace=False)
            aug_flat = aug_flat[idx]

        # Mean shift (per coordinate)
        orig_mean = np.mean(orig_flat, axis=0)
        aug_mean = np.mean(aug_flat, axis=0)
        mean_shift = np.linalg.norm(aug_mean - orig_mean)

        # Variance ratio
        orig_var = np.var(orig_flat, axis=0)
        aug_var = np.var(aug_flat, axis=0)
        variance_ratio = np.mean(aug_var / (orig_var + 1e-10))

        # KS statistic (average across dimensions)
        ks_stats = []
        for d in range(min(orig_flat.shape[1], 10)):  # Sample dimensions
            stat, _ = stats.ks_2samp(orig_flat[:, d], aug_flat[:, d])
            ks_stats.append(stat)
        mean_ks = np.mean(ks_stats)

        # Combined shift score (0 = identical, 1 = very different)
        shift_score = min(1.0, mean_ks)

        return MetricResult(
            name=self.name,
            value=float(shift_score),
            details={
                'mean_shift': float(mean_shift),
                'variance_ratio': float(variance_ratio),
                'ks_statistic': float(mean_ks),
            }
        )


class AugmentationDiversityMetric(BaseMetric):
    """
    Measure diversity among augmented versions of the same sample.

    Given multiple augmented versions of one original sample,
    measures how diverse the augmentations are.
    """

    name = "AugmentationDiversity"
    description = "Measures diversity of augmented samples"

    def compute(
        self,
        data: np.ndarray,
        augmented_versions: Optional[List[np.ndarray]] = None,
        **kwargs,
    ) -> MetricResult:
        """
        Compute augmentation diversity.

        Args:
            data: Original data
            augmented_versions: List of augmented versions of the same sample

        Returns:
            MetricResult with diversity score
        """
        if augmented_versions is None or len(augmented_versions) < 2:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'Need at least 2 augmented versions'}
            )

        # Flatten each version
        flat_versions = [v.flatten() for v in augmented_versions]
        n_versions = len(flat_versions)

        # Pairwise distances
        distances = []
        for i in range(n_versions):
            for j in range(i + 1, n_versions):
                dist = np.linalg.norm(flat_versions[i] - flat_versions[j])
                distances.append(dist)

        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Distance from original
        orig_flat = data.flatten()
        orig_distances = [np.linalg.norm(v - orig_flat) for v in flat_versions]
        mean_orig_distance = np.mean(orig_distances)

        # Diversity score: normalized by data scale
        data_scale = np.std(orig_flat) + 1e-10
        diversity_score = mean_distance / data_scale

        return MetricResult(
            name=self.name,
            value=float(min(1.0, diversity_score)),
            details={
                'mean_pairwise_distance': float(mean_distance),
                'std_pairwise_distance': float(std_distance),
                'mean_distance_from_original': float(mean_orig_distance),
                'num_versions': n_versions,
                'data_scale': float(data_scale),
            }
        )


class CoverageMetric(BaseMetric):
    """
    Measure how well augmentations cover the pose space.

    Uses simple binning to estimate coverage of the pose distribution.
    """

    name = "CoverageMetric"
    description = "Measures augmentation coverage of pose space"

    def __init__(self, num_bins: int = 20):
        """
        Initialize the metric.

        Args:
            num_bins: Number of bins for discretization
        """
        self.num_bins = num_bins

    def compute(
        self,
        data: np.ndarray,
        reference: Optional[np.ndarray] = None,
        **kwargs,
    ) -> MetricResult:
        """
        Compute coverage metric.

        Args:
            data: Augmented data to evaluate
            reference: Optional reference distribution for comparison

        Returns:
            MetricResult with coverage score
        """
        if data.ndim < 3:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'Invalid data shape'}
            )

        # Use first few principal components for coverage estimation
        flat_data = data.reshape(-1, data.shape[-1])

        # PCA to reduce dimensionality
        centered = flat_data - np.mean(flat_data, axis=0)
        try:
            u, s, vh = np.linalg.svd(centered, full_matrices=False)
            # Project to first 3 PCs
            projected = u[:, :3] * s[:3]
        except:
            projected = flat_data[:, :3]

        # Bin counting for coverage
        bins_occupied = set()
        for i in range(3):
            col = projected[:, i]
            min_val, max_val = np.min(col), np.max(col)
            if max_val - min_val < 1e-10:
                continue
            bin_indices = np.floor((col - min_val) / (max_val - min_val + 1e-10) * self.num_bins)
            bin_indices = np.clip(bin_indices, 0, self.num_bins - 1).astype(int)
            bins_occupied.update(zip([i] * len(bin_indices), bin_indices))

        total_bins = 3 * self.num_bins
        coverage = len(bins_occupied) / total_bins

        return MetricResult(
            name=self.name,
            value=float(coverage),
            details={
                'bins_occupied': len(bins_occupied),
                'total_bins': total_bins,
                'num_samples': len(flat_data),
            }
        )
