"""
Base Dataset Classes for KineticAugment Evaluation.

Provides abstract base class for evaluation datasets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np


class EvaluationDataset(ABC):
    """
    Abstract base class for evaluation datasets.

    All evaluation datasets should implement this interface
    for consistent usage across benchmarks.
    """

    name: str = "BaseDataset"

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (landmarks, label)
            - landmarks: np.ndarray of shape (frames, landmarks, 3)
            - label: Integer class label
        """
        pass

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """
        Get list of class names.

        Returns:
            List of class name strings
        """
        pass

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.get_class_names())

    def get_sample_info(self, idx: int) -> dict:
        """
        Get metadata about a sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample metadata
        """
        landmarks, label = self[idx]
        return {
            'index': idx,
            'label': label,
            'class_name': self.get_class_names()[label],
            'num_frames': landmarks.shape[0],
            'num_landmarks': landmarks.shape[1],
        }

    def get_statistics(self) -> dict:
        """
        Compute dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        frame_counts = []
        labels = []

        for i in range(len(self)):
            landmarks, label = self[i]
            frame_counts.append(landmarks.shape[0])
            labels.append(label)

        labels = np.array(labels)
        frame_counts = np.array(frame_counts)

        # Class distribution
        class_counts = np.bincount(labels, minlength=self.num_classes)

        return {
            'num_samples': len(self),
            'num_classes': self.num_classes,
            'mean_frames': float(np.mean(frame_counts)),
            'min_frames': int(np.min(frame_counts)),
            'max_frames': int(np.max(frame_counts)),
            'class_distribution': class_counts.tolist(),
        }


class SyntheticDataset(EvaluationDataset):
    """
    Synthetic dataset for testing without real data.

    Generates random landmark sequences with specified properties.
    """

    name = "SyntheticDataset"

    def __init__(
        self,
        num_samples: int = 100,
        num_classes: int = 10,
        num_frames: int = 30,
        num_landmarks: int = 543,
        seed: Optional[int] = None,
    ):
        """
        Initialize synthetic dataset.

        Args:
            num_samples: Number of samples to generate
            num_classes: Number of classes
            num_frames: Frames per sample
            num_landmarks: Landmarks per frame
            seed: Random seed for reproducibility
        """
        self._num_samples = num_samples
        self._num_classes = num_classes
        self._num_frames = num_frames
        self._num_landmarks = num_landmarks
        self._seed = seed

        self._rng = np.random.RandomState(seed)
        self._labels = self._rng.randint(0, num_classes, size=num_samples)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        # Generate deterministic random data based on index
        rng = np.random.RandomState(self._seed + idx if self._seed else idx)
        landmarks = rng.randn(self._num_frames, self._num_landmarks, 3).astype(np.float32)
        return landmarks, int(self._labels[idx])

    def get_class_names(self) -> List[str]:
        return [f"class_{i}" for i in range(self._num_classes)]
