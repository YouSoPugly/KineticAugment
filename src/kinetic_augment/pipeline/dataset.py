"""
PyTorch Dataset Wrappers for KineticAugment.

Provides Dataset classes for on-the-fly augmentation during training,
with caching support and train/val mode switching.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

# Try to import PyTorch
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a placeholder for type hints
    class Dataset:
        pass

from kinetic_augment.pipeline.pipeline import Pipeline


def check_torch_available() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for dataset wrappers. "
            "Install with: pip install torch"
        )


class AugmentedLandmarkDataset(Dataset):
    """
    PyTorch Dataset with on-the-fly augmentation for landmark data.

    Wraps a data source (list, function, or Dataset) and applies
    augmentation during training. Supports:
    - Train/val mode switching (augment only in train mode)
    - Virtual dataset expansion (num_augmented_versions)
    - Optional caching of augmented samples
    - Label handling

    Example:
        >>> pipeline = Pipeline.from_preset('moderate')
        >>> dataset = AugmentedLandmarkDataset(
        ...     data_source=landmarks_list,
        ...     labels=labels_list,
        ...     pipeline=pipeline,
        ...     mode='train',
        ...     num_augmented_versions=5,
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)

    Attributes:
        data_source: Original data source
        labels: Optional labels
        pipeline: Augmentation pipeline
        mode: 'train' (augment) or 'val' (no augment)
        num_augmented_versions: Virtual size multiplier
    """

    def __init__(
        self,
        data_source: Union[List[np.ndarray], Callable[[int], np.ndarray], 'Dataset'],
        labels: Optional[Union[List, np.ndarray]] = None,
        pipeline: Optional[Pipeline] = None,
        mode: str = 'train',
        num_augmented_versions: int = 1,
        cache_augmentations: bool = False,
        return_tensors: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_source: Source of landmark data:
                - List of np.ndarray (frames, landmarks, 3)
                - Callable that takes index and returns sample
                - Another PyTorch Dataset
            labels: Optional labels corresponding to each sample
            pipeline: Augmentation pipeline (None = no augmentation)
            mode: 'train' (apply augmentation) or 'val' (no augmentation)
            num_augmented_versions: Number of augmented versions per sample.
                Effective dataset size = original_size * num_augmented_versions
            cache_augmentations: If True, cache augmented samples
            return_tensors: If True, return torch.Tensor; else np.ndarray
            seed: Random seed for reproducibility
        """
        check_torch_available()

        self.data_source = data_source
        self.labels = labels
        self.pipeline = pipeline
        self.mode = mode
        self.num_augmented_versions = num_augmented_versions
        self.cache_augmentations = cache_augmentations
        self.return_tensors = return_tensors
        self.seed = seed

        # Determine base length
        if isinstance(data_source, list):
            self._base_length = len(data_source)
        elif hasattr(data_source, '__len__'):
            self._base_length = len(data_source)
        else:
            raise ValueError("data_source must have __len__ or be a list")

        # Cache storage
        self._cache: Dict[int, np.ndarray] = {} if cache_augmentations else None

        # Set seed on pipeline
        if seed is not None and pipeline is not None:
            pipeline.reset_seed(seed)

    def _get_sample(self, base_idx: int) -> np.ndarray:
        """Get a sample from the data source by base index."""
        if isinstance(self.data_source, list):
            return self.data_source[base_idx]
        elif callable(self.data_source):
            return self.data_source(base_idx)
        else:
            # Assume Dataset-like
            item = self.data_source[base_idx]
            if isinstance(item, tuple):
                return item[0]  # Assume (data, label) format
            return item

    def _get_label(self, base_idx: int) -> Optional[Any]:
        """Get label for a sample."""
        if self.labels is None:
            return None
        return self.labels[base_idx]

    def __len__(self) -> int:
        """Return effective dataset length."""
        if self.mode == 'train':
            return self._base_length * self.num_augmented_versions
        else:
            return self._base_length

    def __getitem__(self, idx: int) -> Union[Tuple, np.ndarray, 'torch.Tensor']:
        """
        Get a sample by index.

        In train mode with num_augmented_versions > 1:
        - idx 0 to base_length-1: version 0 of each sample
        - idx base_length to 2*base_length-1: version 1 of each sample
        - etc.

        Returns:
            If labels: (data, label)
            If no labels: data
        """
        # Map virtual index to base index
        if self.mode == 'train' and self.num_augmented_versions > 1:
            version = idx // self._base_length
            base_idx = idx % self._base_length
        else:
            version = 0
            base_idx = idx

        # Check cache
        cache_key = (base_idx, version) if self.mode == 'train' else base_idx
        if self._cache is not None and cache_key in self._cache:
            data = self._cache[cache_key]
        else:
            # Get base sample
            data = self._get_sample(base_idx)

            # Apply augmentation in train mode
            if self.mode == 'train' and self.pipeline is not None:
                data = self.pipeline.process(data)

            # Cache if enabled
            if self._cache is not None:
                self._cache[cache_key] = data

        # Convert to tensor if requested
        if self.return_tensors:
            data = torch.from_numpy(data.astype(np.float32))

        # Get label
        label = self._get_label(base_idx)

        if label is not None:
            if self.return_tensors and not isinstance(label, torch.Tensor):
                if isinstance(label, (int, np.integer)):
                    label = torch.tensor(label)
                elif isinstance(label, np.ndarray):
                    label = torch.from_numpy(label)
            return data, label
        else:
            return data

    def set_mode(self, mode: str) -> 'AugmentedLandmarkDataset':
        """
        Switch between train and val modes.

        Args:
            mode: 'train' or 'val'

        Returns:
            self for method chaining
        """
        if mode not in ['train', 'val']:
            raise ValueError(f"Mode must be 'train' or 'val', got '{mode}'")
        self.mode = mode
        return self

    def train(self) -> 'AugmentedLandmarkDataset':
        """Switch to train mode."""
        return self.set_mode('train')

    def eval(self) -> 'AugmentedLandmarkDataset':
        """Switch to evaluation (val) mode."""
        return self.set_mode('val')

    def clear_cache(self) -> None:
        """Clear the augmentation cache."""
        if self._cache is not None:
            self._cache.clear()

    @property
    def base_length(self) -> int:
        """Return the original dataset length (without augmentation expansion)."""
        return self._base_length


class LandmarkSequenceDataset(Dataset):
    """
    Dataset for variable-length landmark sequences with padding/truncation.

    Handles sequences of different lengths by padding shorter sequences
    and truncating longer ones to a fixed length.

    Example:
        >>> dataset = LandmarkSequenceDataset(
        ...     sequences=sequence_list,
        ...     labels=label_list,
        ...     max_length=100,
        ...     pipeline=Pipeline.from_preset('conservative'),
        ... )
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: Optional[List] = None,
        max_length: Optional[int] = None,
        pipeline: Optional[Pipeline] = None,
        mode: str = 'train',
        padding_value: float = 0.0,
        return_lengths: bool = True,
        return_tensors: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            sequences: List of sequences, each (num_frames, landmarks, 3)
            labels: Optional labels for each sequence
            max_length: Maximum sequence length (None = use longest)
            pipeline: Augmentation pipeline
            mode: 'train' or 'val'
            padding_value: Value to use for padding
            return_lengths: If True, return original lengths
            return_tensors: If True, return torch.Tensor
        """
        check_torch_available()

        self.sequences = sequences
        self.labels = labels
        self.pipeline = pipeline
        self.mode = mode
        self.padding_value = padding_value
        self.return_lengths = return_lengths
        self.return_tensors = return_tensors

        # Compute max length if not specified
        if max_length is None:
            self.max_length = max(seq.shape[0] for seq in sequences)
        else:
            self.max_length = max_length

        # Store original lengths
        self.lengths = [min(seq.shape[0], self.max_length) for seq in sequences]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a padded/truncated sequence.

        Returns:
            If return_lengths: (data, length, label) or (data, length)
            Else: (data, label) or data
        """
        seq = self.sequences[idx]
        orig_length = seq.shape[0]

        # Apply augmentation in train mode (before padding)
        if self.mode == 'train' and self.pipeline is not None:
            seq = self.pipeline.process(seq)

        # Truncate if needed
        if seq.shape[0] > self.max_length:
            seq = seq[:self.max_length]

        # Pad if needed
        actual_length = seq.shape[0]
        if actual_length < self.max_length:
            pad_shape = (self.max_length - actual_length,) + seq.shape[1:]
            padding = np.full(pad_shape, self.padding_value, dtype=seq.dtype)
            seq = np.concatenate([seq, padding], axis=0)

        # Convert to tensor
        if self.return_tensors:
            seq = torch.from_numpy(seq.astype(np.float32))
            actual_length = torch.tensor(actual_length)

        # Get label
        label = self.labels[idx] if self.labels is not None else None
        if label is not None and self.return_tensors:
            if isinstance(label, (int, np.integer)):
                label = torch.tensor(label)
            elif isinstance(label, np.ndarray):
                label = torch.from_numpy(label)

        # Build return tuple
        if self.return_lengths:
            if label is not None:
                return seq, actual_length, label
            else:
                return seq, actual_length
        else:
            if label is not None:
                return seq, label
            else:
                return seq

    def set_mode(self, mode: str) -> 'LandmarkSequenceDataset':
        """Switch between train and val modes."""
        if mode not in ['train', 'val']:
            raise ValueError(f"Mode must be 'train' or 'val', got '{mode}'")
        self.mode = mode
        return self

    def train(self) -> 'LandmarkSequenceDataset':
        """Switch to train mode."""
        return self.set_mode('train')

    def eval(self) -> 'LandmarkSequenceDataset':
        """Switch to evaluation mode."""
        return self.set_mode('val')


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    **kwargs,
) -> Tuple:
    """
    Create train and validation DataLoaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        **kwargs: Additional arguments for DataLoader

    Returns:
        (train_loader,) or (train_loader, val_loader)
    """
    check_torch_available()
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )
        return train_loader, val_loader

    return (train_loader,)
