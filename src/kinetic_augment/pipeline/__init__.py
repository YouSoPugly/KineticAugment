"""
Pipeline Module for KineticAugment.

Provides unified augmentation pipelines with YAML configuration,
PyTorch Dataset wrappers, and configurable constraint enforcement.

Main Components:
- Pipeline: Main augmentation pipeline with constraint enforcement
- AugmentationChain: Composable chain of augmentations
- PipelineConfig: YAML-serializable configuration
- AugmentedLandmarkDataset: PyTorch Dataset with on-the-fly augmentation

Example:
    >>> from kinetic_augment.pipeline import Pipeline, AugmentedLandmarkDataset
    >>>
    >>> # Create pipeline from preset
    >>> pipeline = Pipeline.from_preset('moderate')
    >>>
    >>> # Or from YAML
    >>> pipeline = Pipeline.from_yaml('configs/slr_profile.yaml')
    >>>
    >>> # Single sample
    >>> augmented = pipeline.process(landmarks)
    >>>
    >>> # With PyTorch DataLoader
    >>> dataset = AugmentedLandmarkDataset(
    ...     data_source=landmarks_list,
    ...     labels=labels_list,
    ...     pipeline=pipeline,
    ...     mode='train',
    ... )
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

from kinetic_augment.pipeline.config import PipelineConfig, ConstraintConfig
from kinetic_augment.pipeline.chain import AugmentationChain, create_chain_from_preset
from kinetic_augment.pipeline.pipeline import Pipeline, list_presets
from kinetic_augment.pipeline.presets import get_preset, PRESET_NAMES

# PyTorch datasets (optional, requires torch)
try:
    from kinetic_augment.pipeline.dataset import (
        AugmentedLandmarkDataset,
        LandmarkSequenceDataset,
        create_data_loaders,
        TORCH_AVAILABLE,
    )
except ImportError:
    TORCH_AVAILABLE = False
    AugmentedLandmarkDataset = None
    LandmarkSequenceDataset = None
    create_data_loaders = None

__all__ = [
    # Configuration
    'PipelineConfig',
    'ConstraintConfig',
    # Chain
    'AugmentationChain',
    'create_chain_from_preset',
    # Pipeline
    'Pipeline',
    # Presets
    'get_preset',
    'list_presets',
    'PRESET_NAMES',
    # Datasets (optional)
    'AugmentedLandmarkDataset',
    'LandmarkSequenceDataset',
    'create_data_loaders',
    'TORCH_AVAILABLE',
]
