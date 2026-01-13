"""
Collision Detection Package for KineticAugment.

Provides PyBullet-based self-collision detection and resolution
for SMPL-X body models.

Components:
- CollisionConstraint: Main constraint class (check/enforce interface)
- CollisionChecker: PyBullet-based collision detection
- CollisionViolation: Extended violation class with collision info
- GradientResolver: Push-based collision resolution

Requirements:
- PyBullet (optional): pip install pybullet

Example:
    >>> from kinetic_augment.constraints.collision import CollisionConstraint
    >>> constraint = CollisionConstraint(smplx_wrapper)
    >>> violations = constraint.check(params)
    >>> corrected = constraint.enforce(params)
"""

from kinetic_augment.constraints.collision.collision_constraint import (
    CollisionConstraint,
    CollisionViolation,
    check_pybullet_available,
)

from kinetic_augment.constraints.collision.body_parts import (
    COLLISION_BODY_PARTS,
    SLR_COLLISION_PAIRS,
    FULL_COLLISION_PAIRS,
    ADJACENT_PARTS,
    CONTROLLING_JOINTS,
    get_body_part_params,
    get_capsule_params,
    get_sphere_params,
    are_adjacent,
    get_contact_tolerance,
)

# Conditionally export PyBullet-dependent classes
try:
    from kinetic_augment.constraints.collision.collision_checker import (
        CollisionChecker,
        TieredCollisionChecker,
        ContactPoint,
        PYBULLET_AVAILABLE,
    )
    from kinetic_augment.constraints.collision.resolution import (
        CollisionResolver,
        GradientResolver,
        RejectionResolver,
        InterpolationResolver,
        create_resolver,
    )
except ImportError:
    PYBULLET_AVAILABLE = False
    CollisionChecker = None
    TieredCollisionChecker = None
    ContactPoint = None
    CollisionResolver = None
    GradientResolver = None
    RejectionResolver = None
    InterpolationResolver = None
    create_resolver = None


__all__ = [
    # Main constraint class
    'CollisionConstraint',
    'CollisionViolation',
    'check_pybullet_available',

    # Body part definitions
    'COLLISION_BODY_PARTS',
    'SLR_COLLISION_PAIRS',
    'FULL_COLLISION_PAIRS',
    'ADJACENT_PARTS',
    'CONTROLLING_JOINTS',

    # Utility functions
    'get_body_part_params',
    'get_capsule_params',
    'get_sphere_params',
    'are_adjacent',
    'get_contact_tolerance',

    # PyBullet-dependent (may be None)
    'CollisionChecker',
    'TieredCollisionChecker',
    'ContactPoint',
    'PYBULLET_AVAILABLE',
    'CollisionResolver',
    'GradientResolver',
    'RejectionResolver',
    'InterpolationResolver',
    'create_resolver',
]
