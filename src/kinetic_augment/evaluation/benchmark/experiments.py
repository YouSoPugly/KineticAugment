"""
Predefined Experiment Configurations for KineticAugment Evaluation.

Provides standard experiment setups for ablation studies.
"""

from kinetic_augment.evaluation.benchmark.runner import ExperimentConfig


# Preset comparison experiments
PRESET_COMPARISON = [
    ExperimentConfig(
        name='baseline_none',
        description='No augmentation baseline',
        pipeline_config='none',
    ),
    ExperimentConfig(
        name='preset_conservative',
        description='Conservative augmentation preset',
        pipeline_config='conservative',
    ),
    ExperimentConfig(
        name='preset_moderate',
        description='Moderate augmentation preset',
        pipeline_config='moderate',
    ),
    ExperimentConfig(
        name='preset_aggressive',
        description='Aggressive augmentation preset',
        pipeline_config='aggressive',
    ),
]


# Individual augmentation ablation
INDIVIDUAL_AUGMENTATIONS = [
    ExperimentConfig(
        name='rotation_only',
        description='GlobalRotation only',
        pipeline_config={
            'mode': 'landmark',
            'plan': [
                {'operation': 'GlobalRotation', 'probability': 1.0,
                 'params': {'max_angle_deg': {'x': 15, 'y': 15, 'z': 10}}},
            ],
        },
    ),
    ExperimentConfig(
        name='scaling_only',
        description='GlobalScaling only',
        pipeline_config={
            'mode': 'landmark',
            'plan': [
                {'operation': 'GlobalScaling', 'probability': 1.0,
                 'params': {'scale_range': [0.9, 1.1]}},
            ],
        },
    ),
    ExperimentConfig(
        name='noise_only',
        description='GaussianNoise only',
        pipeline_config={
            'mode': 'landmark',
            'plan': [
                {'operation': 'GaussianNoise', 'probability': 1.0,
                 'params': {'stddev': 0.01}},
            ],
        },
    ),
    ExperimentConfig(
        name='timewarping_only',
        description='TimeWarping only',
        pipeline_config={
            'mode': 'landmark',
            'plan': [
                {'operation': 'TimeWarping', 'probability': 1.0,
                 'params': {'max_warp_factor': 0.2}},
            ],
        },
    ),
    ExperimentConfig(
        name='jittering_only',
        description='TrajectoryJittering only',
        pipeline_config={
            'mode': 'landmark',
            'plan': [
                {'operation': 'TrajectoryJittering', 'probability': 1.0,
                 'params': {'amplitude': 0.02, 'frequency': 0.5}},
            ],
        },
    ),
]


# Dataset size experiments
def create_augmentation_factor_experiments(
    preset: str = 'moderate',
    factors: list = None,
) -> list:
    """
    Create experiments varying augmentation factor.

    Args:
        preset: Base preset to use
        factors: List of num_augmented_versions to test

    Returns:
        List of ExperimentConfig
    """
    if factors is None:
        factors = [1, 2, 5, 10]

    return [
        ExperimentConfig(
            name=f'{preset}_x{factor}',
            description=f'{preset.capitalize()} preset with {factor}x augmentation',
            pipeline_config=preset,
            num_augmented_versions=factor,
        )
        for factor in factors
    ]


# Constraint ablation
CONSTRAINT_ABLATION = [
    ExperimentConfig(
        name='moderate_no_constraints',
        description='Moderate without constraint enforcement',
        pipeline_config={
            'mode': 'landmark',
            'plan': [
                {'operation': 'GlobalRotation', 'probability': 0.6,
                 'params': {'max_angle_deg': {'x': 10, 'y': 15, 'z': 5}}},
                {'operation': 'GlobalScaling', 'probability': 0.4,
                 'params': {'scale_range': [0.9, 1.1]}},
                {'operation': 'GaussianNoise', 'probability': 0.5,
                 'params': {'stddev': 0.005}},
            ],
            'constraints': {
                'joint_limits': False,
                'velocity_limits': False,
            },
        },
    ),
    ExperimentConfig(
        name='moderate_joint_limits',
        description='Moderate with joint limits only',
        pipeline_config={
            'mode': 'landmark',
            'plan': [
                {'operation': 'GlobalRotation', 'probability': 0.6,
                 'params': {'max_angle_deg': {'x': 10, 'y': 15, 'z': 5}}},
                {'operation': 'GlobalScaling', 'probability': 0.4,
                 'params': {'scale_range': [0.9, 1.1]}},
                {'operation': 'GaussianNoise', 'probability': 0.5,
                 'params': {'stddev': 0.005}},
            ],
            'constraints': {
                'joint_limits': True,
                'velocity_limits': False,
            },
        },
    ),
    ExperimentConfig(
        name='moderate_all_constraints',
        description='Moderate with all constraints',
        pipeline_config={
            'mode': 'landmark',
            'plan': [
                {'operation': 'GlobalRotation', 'probability': 0.6,
                 'params': {'max_angle_deg': {'x': 10, 'y': 15, 'z': 5}}},
                {'operation': 'GlobalScaling', 'probability': 0.4,
                 'params': {'scale_range': [0.9, 1.1]}},
                {'operation': 'GaussianNoise', 'probability': 0.5,
                 'params': {'stddev': 0.005}},
            ],
            'constraints': {
                'joint_limits': True,
                'velocity_limits': True,
            },
        },
    ),
]


# Quick test experiments (for development)
QUICK_TEST = [
    ExperimentConfig(
        name='quick_baseline',
        description='Quick baseline test',
        pipeline_config='none',
        epochs=5,
        batch_size=64,
    ),
    ExperimentConfig(
        name='quick_moderate',
        description='Quick moderate test',
        pipeline_config='moderate',
        epochs=5,
        batch_size=64,
    ),
]


def get_standard_experiments(
    include_individual: bool = True,
    include_constraint: bool = True,
) -> list:
    """
    Get standard set of experiments for comprehensive evaluation.

    Args:
        include_individual: Include individual augmentation ablation
        include_constraint: Include constraint ablation

    Returns:
        List of ExperimentConfig
    """
    experiments = list(PRESET_COMPARISON)

    if include_individual:
        experiments.extend(INDIVIDUAL_AUGMENTATIONS)

    if include_constraint:
        experiments.extend(CONSTRAINT_ABLATION)

    return experiments
