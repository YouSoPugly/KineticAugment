"""
Built-in Augmentation Presets for KineticAugment.

Provides predefined augmentation configurations for common use cases.
"""

from typing import List

from kinetic_augment.pipeline.config import PipelineConfig, ConstraintConfig


# Available preset names
PRESET_NAMES: List[str] = ['none', 'conservative', 'moderate', 'aggressive']


def get_preset(name: str) -> PipelineConfig:
    """
    Get a predefined augmentation preset.

    Args:
        name: Preset name - 'none', 'conservative', 'moderate', 'aggressive'

    Returns:
        PipelineConfig with preset values

    Raises:
        ValueError: If preset name is unknown
    """
    presets = {
        'none': _preset_none,
        'conservative': _preset_conservative,
        'moderate': _preset_moderate,
        'aggressive': _preset_aggressive,
    }

    if name not in presets:
        raise ValueError(
            f"Unknown preset: '{name}'. Available presets: {PRESET_NAMES}"
        )

    return presets[name]()


def _preset_none() -> PipelineConfig:
    """No augmentation, constraint enforcement only."""
    return PipelineConfig(
        profile_name="none",
        description="No augmentation, constraint enforcement only",
        plan=[],
        constraints=ConstraintConfig(
            joint_limits=True,
            velocity_limits=True,
            collision_detection=False,
        ),
    )


def _preset_conservative() -> PipelineConfig:
    """
    Conservative augmentations suitable for testing and validation.

    Light transformations that preserve sign semantics well.
    Good for initial experiments or when data quality is uncertain.
    """
    return PipelineConfig(
        profile_name="conservative",
        description="Light augmentations preserving sign semantics",
        plan=[
            {
                'operation': 'GlobalRotation',
                'probability': 0.5,
                'params': {'max_angle_deg': {'x': 5, 'y': 8, 'z': 3}},
            },
            {
                'operation': 'GlobalScaling',
                'probability': 0.3,
                'params': {'scale_range': [0.95, 1.05]},
            },
            {
                'operation': 'GaussianNoise',
                'probability': 0.3,
                'params': {
                    'stddev': 0.002,
                    'landmark_groups': ['pose'],
                },
            },
        ],
        constraints=ConstraintConfig(
            joint_limits=True,
            velocity_limits=True,
            collision_detection=False,
            use_sign_language_limits=True,
        ),
    )


def _preset_moderate() -> PipelineConfig:
    """
    Moderate augmentations for typical training.

    Balanced transformations suitable for most SLR training scenarios.
    Recommended starting point for production training.
    """
    return PipelineConfig(
        profile_name="moderate",
        description="Balanced augmentations for SLR training",
        plan=[
            {
                'operation': 'GlobalRotation',
                'probability': 0.6,
                'params': {'max_angle_deg': {'x': 10, 'y': 15, 'z': 5}},
            },
            {
                'operation': 'GlobalScaling',
                'probability': 0.4,
                'params': {'scale_range': [0.9, 1.1]},
            },
            {
                'operation': 'TrajectoryJittering',
                'probability': 0.3,
                'params': {
                    'amplitude': 0.01,
                    'frequency': 0.3,
                },
            },
            {
                'operation': 'TimeWarping',
                'probability': 0.4,
                'params': {'max_warp_factor': 0.15},
            },
            {
                'operation': 'GaussianNoise',
                'probability': 0.5,
                'params': {
                    'stddev': 0.005,
                    'landmark_groups': ['pose', 'left_hand', 'right_hand'],
                },
            },
        ],
        constraints=ConstraintConfig(
            joint_limits=True,
            velocity_limits=True,
            collision_detection=False,
            use_sign_language_limits=True,
        ),
    )


def _preset_aggressive() -> PipelineConfig:
    """
    Aggressive augmentations for maximum diversity.

    Strong transformations for data-limited scenarios where
    maximizing diversity is more important than preserving
    exact motion characteristics.
    """
    return PipelineConfig(
        profile_name="aggressive",
        description="Strong augmentations for data-limited scenarios",
        plan=[
            {
                'operation': 'GlobalRotation',
                'probability': 0.8,
                'params': {'max_angle_deg': {'x': 15, 'y': 25, 'z': 10}},
            },
            {
                'operation': 'GlobalScaling',
                'probability': 0.6,
                'params': {'scale_range': [0.85, 1.15]},
            },
            {
                'operation': 'PoseFlipping',
                'probability': 0.3,
                'params': {},
            },
            {
                'operation': 'TrajectoryJittering',
                'probability': 0.5,
                'params': {
                    'amplitude': 0.02,
                    'frequency': 0.5,
                },
            },
            {
                'operation': 'TimeWarping',
                'probability': 0.6,
                'params': {
                    'max_warp_factor': 0.25,
                    'num_knots': 5,
                },
            },
            {
                'operation': 'SpeedVariation',
                'probability': 0.4,
                'params': {
                    'speed_range': [0.7, 1.3],
                    'output_frames': 'same',
                },
            },
            {
                'operation': 'GaussianNoise',
                'probability': 0.7,
                'params': {
                    'stddev': 0.01,
                    'landmark_groups': ['pose', 'left_hand', 'right_hand'],
                },
            },
        ],
        constraints=ConstraintConfig(
            joint_limits=True,
            velocity_limits=True,
            collision_detection=False,
            use_sign_language_limits=True,
        ),
    )


def list_presets() -> List[str]:
    """Get list of available preset names."""
    return PRESET_NAMES.copy()
