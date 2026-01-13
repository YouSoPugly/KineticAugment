"""
Augmentation Chain for KineticAugment.

Provides composable augmentation chains with probability-based execution,
type validation, and reproducibility support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np

from kinetic_augment.augmentations.base import (
    BaseAugmentation,
    LandmarkAugmentation,
    SMPLXAugmentation,
    TemporalAugmentation,
    AugmentationRegistry,
)


@dataclass
class ChainStep:
    """
    A single step in the augmentation chain.

    Attributes:
        augmentation: The augmentation instance
        probability: Probability of applying this step (0-1)
        name: Operation name for logging/debugging
    """
    augmentation: BaseAugmentation
    probability: float = 1.0
    name: str = ""

    def __post_init__(self):
        if not 0 <= self.probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {self.probability}")
        if not self.name:
            self.name = self.augmentation.__class__.__name__


class AugmentationChain:
    """
    Composable chain of augmentations with probability-based execution.

    The chain applies augmentations sequentially, with each step having
    an independent probability of being applied. Supports both landmark
    and SMPL-X modes with type validation.

    Example:
        >>> chain = AugmentationChain(mode='landmark', seed=42)
        >>> chain.add('GlobalRotation', probability=0.6, config={'max_angle_deg': {'x': 10}})
        >>> chain.add('GaussianNoise', probability=0.3, config={'stddev': 0.005})
        >>> augmented = chain(landmarks)

    Attributes:
        mode: Operating mode - 'landmark' or 'smplx'
        seed: Random seed for reproducibility
        steps: List of ChainStep objects in the chain
    """

    def __init__(
        self,
        mode: str = 'landmark',
        seed: Optional[int] = None,
    ):
        """
        Initialize the augmentation chain.

        Args:
            mode: Operating mode - 'landmark' or 'smplx'
            seed: Random seed for reproducibility
        """
        if mode not in ['landmark', 'smplx']:
            raise ValueError(f"Mode must be 'landmark' or 'smplx', got '{mode}'")

        self.mode = mode
        self.seed = seed
        self.steps: List[ChainStep] = []
        self._rng = np.random.RandomState(seed)

    def add(
        self,
        operation: Union[str, BaseAugmentation],
        probability: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> 'AugmentationChain':
        """
        Add an augmentation to the chain.

        Args:
            operation: Augmentation name (string) or instance
            probability: Probability of applying this augmentation
            config: Configuration dict (only if operation is a string)

        Returns:
            self for method chaining

        Raises:
            ValueError: If augmentation is incompatible with chain mode
        """
        # Get or create augmentation instance
        if isinstance(operation, str):
            augmentation = AugmentationRegistry.create(operation, config or {})
            name = operation
        else:
            augmentation = operation
            name = operation.__class__.__name__

        # Validate mode compatibility
        self._validate_compatibility(augmentation)

        step = ChainStep(
            augmentation=augmentation,
            probability=probability,
            name=name,
        )
        self.steps.append(step)
        return self

    def _validate_compatibility(self, augmentation: BaseAugmentation) -> None:
        """
        Check if augmentation is compatible with chain mode.

        Args:
            augmentation: The augmentation to validate

        Raises:
            ValueError: If augmentation is incompatible
        """
        # Temporal augmentations work with both modes
        if isinstance(augmentation, TemporalAugmentation):
            return

        # Check mode-specific compatibility
        if self.mode == 'landmark':
            if isinstance(augmentation, SMPLXAugmentation):
                raise ValueError(
                    f"Cannot add SMPL-X augmentation '{augmentation.__class__.__name__}' "
                    f"to landmark-mode chain. Use mode='smplx' or choose a "
                    f"LandmarkAugmentation instead."
                )
        elif self.mode == 'smplx':
            if isinstance(augmentation, LandmarkAugmentation):
                # LandmarkAugmentation can work in SMPL-X mode too
                # (applied after FK to landmarks)
                pass

    def __call__(
        self,
        data: Union[np.ndarray, Dict[str, np.ndarray]],
        **kwargs,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Apply the augmentation chain to input data.

        Args:
            data: Input data (landmarks array or SMPL-X params dict)
            **kwargs: Additional arguments passed to augmentations

        Returns:
            Augmented data in the same format as input
        """
        result = data

        for step in self.steps:
            # Check probability
            if self._rng.random() > step.probability:
                continue

            # Validate data for this augmentation
            if not step.augmentation.validate(result):
                # Skip incompatible augmentations silently
                continue

            # Apply augmentation
            result = step.augmentation.apply(result, **kwargs)

        return result

    def apply_deterministic(
        self,
        data: Union[np.ndarray, Dict[str, np.ndarray]],
        apply_mask: Optional[List[bool]] = None,
        **kwargs,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Apply chain with explicit control over which steps are applied.

        Args:
            data: Input data
            apply_mask: List of booleans for each step (True = apply)
            **kwargs: Additional arguments passed to augmentations

        Returns:
            Augmented data
        """
        if apply_mask is None:
            apply_mask = [True] * len(self.steps)

        if len(apply_mask) != len(self.steps):
            raise ValueError(
                f"apply_mask length ({len(apply_mask)}) must match "
                f"number of steps ({len(self.steps)})"
            )

        result = data
        for step, should_apply in zip(self.steps, apply_mask):
            if not should_apply:
                continue
            if not step.augmentation.validate(result):
                continue
            result = step.augmentation.apply(result, **kwargs)

        return result

    def reset_rng(self, seed: Optional[int] = None) -> None:
        """
        Reset the random number generator.

        Args:
            seed: New seed (uses original seed if None)
        """
        if seed is not None:
            self.seed = seed
        self._rng = np.random.RandomState(self.seed)

    @classmethod
    def from_config(
        cls,
        config: List[Dict[str, Any]],
        mode: str = 'landmark',
        seed: Optional[int] = None,
    ) -> 'AugmentationChain':
        """
        Create chain from configuration list.

        Args:
            config: List of step configurations, each with:
                - operation: Augmentation name
                - probability: Application probability (default 1.0)
                - params: Augmentation parameters (default {})
            mode: Operating mode
            seed: Random seed

        Returns:
            Configured AugmentationChain

        Example config:
            [
                {'operation': 'GlobalRotation', 'probability': 0.6,
                 'params': {'max_angle_deg': {'x': 10}}},
                {'operation': 'GaussianNoise', 'probability': 0.3,
                 'params': {'stddev': 0.005}},
            ]
        """
        chain = cls(mode=mode, seed=seed)

        for step_config in config:
            operation = step_config.get('operation')
            if operation is None:
                raise ValueError("Each step must have an 'operation' key")

            probability = step_config.get('probability', 1.0)
            params = step_config.get('params', {})

            chain.add(operation, probability=probability, config=params)

        return chain

    def get_step_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all steps in the chain.

        Returns:
            List of dicts with step information
        """
        return [
            {
                'name': step.name,
                'probability': step.probability,
                'category': step.augmentation.category,
                'requires_smplx': step.augmentation.requires_smplx,
            }
            for step in self.steps
        ]

    def __len__(self) -> int:
        """Return number of steps in the chain."""
        return len(self.steps)

    def __repr__(self) -> str:
        step_names = [s.name for s in self.steps]
        return f"AugmentationChain(mode={self.mode}, steps={step_names})"


def create_chain_from_preset(
    preset_name: str,
    seed: Optional[int] = None,
) -> AugmentationChain:
    """
    Create an augmentation chain from a preset.

    Args:
        preset_name: Name of the preset
        seed: Random seed for reproducibility

    Returns:
        Configured AugmentationChain
    """
    from kinetic_augment.pipeline.presets import get_preset

    config = get_preset(preset_name)
    return AugmentationChain.from_config(
        config=config.plan,
        mode=config.mode,
        seed=seed if seed is not None else config.seed,
    )
