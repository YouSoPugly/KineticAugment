"""
Base classes for KineticAugment Evaluation Module.

Provides core infrastructure for metrics computation and aggregation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class MetricResult:
    """
    Result from a metric computation.

    Attributes:
        name: Metric name
        value: Primary metric value (typically 0-1 for normalized metrics)
        details: Additional metric details and sub-metrics
    """
    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"MetricResult({self.name}={self.value:.4f})"


class BaseMetric(ABC):
    """
    Abstract base class for all metrics.

    All metrics must implement the `compute()` method which takes
    input data and returns a MetricResult.

    Attributes:
        name: Human-readable metric name
        description: Brief description of what the metric measures
    """

    name: str = "BaseMetric"
    description: str = ""

    @abstractmethod
    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """
        Compute the metric on input data.

        Args:
            data: Input data (typically landmarks of shape (frames, landmarks, 3))
            **kwargs: Additional arguments specific to the metric

        Returns:
            MetricResult with computed value and details
        """
        pass

    def compute_batch(
        self,
        batch: List[np.ndarray],
        **kwargs,
    ) -> List[MetricResult]:
        """
        Compute metric on a batch of samples.

        Args:
            batch: List of input samples
            **kwargs: Additional arguments

        Returns:
            List of MetricResult for each sample
        """
        return [self.compute(sample, **kwargs) for sample in batch]

    def aggregate(self, results: List[MetricResult]) -> MetricResult:
        """
        Aggregate multiple results into a single summary result.

        Args:
            results: List of MetricResult to aggregate

        Returns:
            Aggregated MetricResult with mean value
        """
        if not results:
            return MetricResult(name=self.name, value=0.0)

        values = [r.value for r in results]
        return MetricResult(
            name=self.name,
            value=float(np.mean(values)),
            details={
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values),
            }
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MetricAggregator:
    """
    Aggregates multiple metrics into a unified evaluation.

    Runs multiple metrics on input data and collects results into
    a single dictionary.

    Example:
        >>> aggregator = MetricAggregator([
        ...     JointValidityMetric(),
        ...     SmoothnessMetric(),
        ... ])
        >>> results = aggregator.evaluate(landmarks)
        >>> print(results['JointValidityMetric'].value)
    """

    def __init__(self, metrics: List[BaseMetric]):
        """
        Initialize the aggregator.

        Args:
            metrics: List of metric instances to run
        """
        self.metrics = metrics

    def evaluate(
        self,
        data: Union[np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Dict[str, MetricResult]:
        """
        Run all metrics on input data.

        Args:
            data: Single sample or list of samples
            **kwargs: Additional arguments passed to all metrics

        Returns:
            Dictionary mapping metric names to results
        """
        results = {}

        for metric in self.metrics:
            try:
                if isinstance(data, list):
                    # Batch mode: compute on each and aggregate
                    batch_results = metric.compute_batch(data, **kwargs)
                    result = metric.aggregate(batch_results)
                else:
                    # Single sample mode
                    result = metric.compute(data, **kwargs)
                results[metric.name] = result
            except Exception as e:
                # Store error in result
                results[metric.name] = MetricResult(
                    name=metric.name,
                    value=float('nan'),
                    details={'error': str(e)}
                )

        return results

    def evaluate_comparison(
        self,
        original: Union[np.ndarray, List[np.ndarray]],
        augmented: Union[np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Dict[str, Dict[str, MetricResult]]:
        """
        Evaluate both original and augmented data for comparison.

        Args:
            original: Original data samples
            augmented: Augmented data samples
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'original' and 'augmented' metric results
        """
        return {
            'original': self.evaluate(original, **kwargs),
            'augmented': self.evaluate(augmented, **kwargs),
        }

    def get_summary(self, results: Dict[str, MetricResult]) -> Dict[str, float]:
        """
        Get simple value-only summary of results.

        Args:
            results: Dictionary of MetricResults

        Returns:
            Dictionary mapping metric names to values
        """
        return {name: result.value for name, result in results.items()}

    def __repr__(self) -> str:
        metric_names = [m.name for m in self.metrics]
        return f"MetricAggregator(metrics={metric_names})"


class ComparisonMetric(BaseMetric):
    """
    Base class for metrics that compare two samples.

    Used for metrics like distribution shift that need both
    original and augmented data.
    """

    @abstractmethod
    def compute_comparison(
        self,
        original: np.ndarray,
        augmented: np.ndarray,
        **kwargs,
    ) -> MetricResult:
        """
        Compute comparison metric between original and augmented.

        Args:
            original: Original data
            augmented: Augmented data
            **kwargs: Additional arguments

        Returns:
            MetricResult with comparison value
        """
        pass

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """
        For comparison metrics, compute() requires 'original' in kwargs.
        """
        original = kwargs.get('original')
        if original is None:
            return MetricResult(
                name=self.name,
                value=float('nan'),
                details={'error': 'Comparison metric requires original data'}
            )
        return self.compute_comparison(original, data, **kwargs)
