"""
Body Part Definitions for Collision Detection.

Defines body part segmentation for building collision geometry from SMPL-X joints.
Each body part is represented by a simple collision primitive (capsule or sphere)
derived from joint positions.
"""

from typing import Dict, List, Tuple, Set
import numpy as np

# =============================================================================
# Body Part Definitions
# =============================================================================

# Body parts with their defining joints and geometry type
# Format: {part_name: {'joints': [joint_names], 'type': 'capsule'|'sphere', 'radius_scale': float}}
# radius_scale is relative to body height (approximately 1.7m for SMPL-X neutral)
COLLISION_BODY_PARTS: Dict[str, Dict] = {
    # Torso - large compound capsule from pelvis to spine3
    'torso': {
        'joints': ['pelvis', 'spine3'],
        'type': 'capsule',
        'radius_scale': 0.12,  # ~20cm radius
    },

    # Head - sphere at head joint
    'head': {
        'joints': ['head'],
        'type': 'sphere',
        'radius_scale': 0.09,  # ~15cm radius
    },

    # Upper arms (shoulder to elbow)
    'left_upper_arm': {
        'joints': ['left_shoulder', 'left_elbow'],
        'type': 'capsule',
        'radius_scale': 0.04,  # ~7cm radius
    },
    'right_upper_arm': {
        'joints': ['right_shoulder', 'right_elbow'],
        'type': 'capsule',
        'radius_scale': 0.04,
    },

    # Forearms (elbow to wrist)
    'left_forearm': {
        'joints': ['left_elbow', 'left_wrist'],
        'type': 'capsule',
        'radius_scale': 0.035,  # ~6cm radius
    },
    'right_forearm': {
        'joints': ['right_elbow', 'right_wrist'],
        'type': 'capsule',
        'radius_scale': 0.035,
    },

    # Hands - spheres at wrist (simplified)
    'left_hand': {
        'joints': ['left_wrist'],
        'type': 'sphere',
        'radius_scale': 0.05,  # ~8.5cm radius
    },
    'right_hand': {
        'joints': ['right_wrist'],
        'type': 'sphere',
        'radius_scale': 0.05,
    },

    # Upper legs (hip to knee)
    'left_thigh': {
        'joints': ['left_hip', 'left_knee'],
        'type': 'capsule',
        'radius_scale': 0.06,  # ~10cm radius
    },
    'right_thigh': {
        'joints': ['right_hip', 'right_knee'],
        'type': 'capsule',
        'radius_scale': 0.06,
    },

    # Lower legs (knee to ankle)
    'left_calf': {
        'joints': ['left_knee', 'left_ankle'],
        'type': 'capsule',
        'radius_scale': 0.045,  # ~7.5cm radius
    },
    'right_calf': {
        'joints': ['right_knee', 'right_ankle'],
        'type': 'capsule',
        'radius_scale': 0.045,
    },
}

# =============================================================================
# Collision Pair Definitions
# =============================================================================

# SLR-specific collision pairs to check
# Focused on hand/arm interactions common in signing
SLR_COLLISION_PAIRS: List[Tuple[str, str]] = [
    # Hand-to-hand (two-handed signs)
    ('left_hand', 'right_hand'),

    # Hand-to-torso (signs touching chest, "ME", "FEEL", etc.)
    ('left_hand', 'torso'),
    ('right_hand', 'torso'),

    # Hand-to-head (face-touching signs)
    ('left_hand', 'head'),
    ('right_hand', 'head'),

    # Forearm-to-torso (crossed arms, arm against body)
    ('left_forearm', 'torso'),
    ('right_forearm', 'torso'),

    # Upper arm-to-torso (arm pressed against body)
    ('left_upper_arm', 'torso'),
    ('right_upper_arm', 'torso'),

    # Arm-to-arm (crossed arms)
    ('left_forearm', 'right_forearm'),
    ('left_upper_arm', 'right_upper_arm'),
    ('left_forearm', 'right_upper_arm'),
    ('right_forearm', 'left_upper_arm'),
]

# Full body collision pairs (for non-SLR use cases)
FULL_COLLISION_PAIRS: List[Tuple[str, str]] = SLR_COLLISION_PAIRS + [
    # Leg-to-leg
    ('left_thigh', 'right_thigh'),
    ('left_calf', 'right_calf'),

    # Hand-to-leg (reaching down)
    ('left_hand', 'left_thigh'),
    ('right_hand', 'right_thigh'),

    # Head-to-limbs
    ('head', 'left_upper_arm'),
    ('head', 'right_upper_arm'),
]

# =============================================================================
# Adjacent Body Parts (Skip Collision Check)
# =============================================================================

# Body parts that are connected and naturally overlap at joints
# Format: frozenset of (part_a, part_b) pairs
ADJACENT_PARTS: Set[frozenset] = {
    # Arm chain
    frozenset(('left_upper_arm', 'left_forearm')),
    frozenset(('left_forearm', 'left_hand')),
    frozenset(('right_upper_arm', 'right_forearm')),
    frozenset(('right_forearm', 'right_hand')),

    # Leg chain
    frozenset(('left_thigh', 'left_calf')),
    frozenset(('right_thigh', 'right_calf')),

    # Torso connections
    frozenset(('torso', 'head')),
    frozenset(('torso', 'left_upper_arm')),
    frozenset(('torso', 'right_upper_arm')),
    frozenset(('torso', 'left_thigh')),
    frozenset(('torso', 'right_thigh')),
}

# =============================================================================
# SLR-Specific Tolerances
# =============================================================================

# Some signs involve intentional contact - allow slight overlap
# Values are in meters (penetration depth allowed)
SLR_CONTACT_TOLERANCES: Dict[Tuple[str, str], float] = {
    # Hand touching chest (many signs involve this)
    ('left_hand', 'torso'): 0.02,
    ('right_hand', 'torso'): 0.02,

    # Hand near face (common in emotional signs)
    ('left_hand', 'head'): 0.015,
    ('right_hand', 'head'): 0.015,

    # Hands together (compound signs)
    ('left_hand', 'right_hand'): 0.01,
}

# Default tolerance for unlisted pairs
DEFAULT_CONTACT_TOLERANCE: float = 0.005  # 5mm

# =============================================================================
# Controlling Joints for Resolution
# =============================================================================

# Maps body parts to the joints that control their position
# Used by gradient-based resolution to know which joints to adjust
CONTROLLING_JOINTS: Dict[str, List[str]] = {
    'torso': ['spine1', 'spine2', 'spine3'],
    'head': ['neck', 'head'],
    'left_upper_arm': ['left_shoulder'],
    'right_upper_arm': ['right_shoulder'],
    'left_forearm': ['left_shoulder', 'left_elbow'],
    'right_forearm': ['right_shoulder', 'right_elbow'],
    'left_hand': ['left_shoulder', 'left_elbow', 'left_wrist'],
    'right_hand': ['right_shoulder', 'right_elbow', 'right_wrist'],
    'left_thigh': ['left_hip'],
    'right_thigh': ['right_hip'],
    'left_calf': ['left_hip', 'left_knee'],
    'right_calf': ['right_hip', 'right_knee'],
}

# =============================================================================
# Utility Functions
# =============================================================================

# Reference body height for SMPL-X neutral model (meters)
REFERENCE_BODY_HEIGHT: float = 1.7


def get_capsule_params(
    joint_positions: np.ndarray,
    joint_indices: Dict[str, int],
    part_name: str,
    body_height: float = REFERENCE_BODY_HEIGHT,
) -> Dict:
    """
    Compute capsule parameters from joint positions.

    Args:
        joint_positions: Array of shape (num_joints, 3) with joint positions
        joint_indices: Dict mapping joint names to indices
        part_name: Name of the body part
        body_height: Estimated body height for scaling

    Returns:
        Dict with 'center', 'orientation', 'radius', 'height' keys
    """
    config = COLLISION_BODY_PARTS[part_name]
    joints = config['joints']

    if len(joints) != 2:
        raise ValueError(f"Capsule requires exactly 2 joints, got {len(joints)}")

    # Get joint positions
    j1_idx = joint_indices[joints[0]]
    j2_idx = joint_indices[joints[1]]

    p1 = joint_positions[j1_idx]
    p2 = joint_positions[j2_idx]

    # Capsule center is midpoint
    center = (p1 + p2) / 2

    # Direction vector (axis of capsule)
    direction = p2 - p1
    height = np.linalg.norm(direction)
    if height > 1e-6:
        direction = direction / height
    else:
        direction = np.array([0.0, 1.0, 0.0])  # Default up direction

    # Compute quaternion from direction (align Y-axis with direction)
    orientation = _direction_to_quaternion(direction)

    # Scale radius by body height
    radius = config['radius_scale'] * body_height

    return {
        'center': center,
        'orientation': orientation,
        'radius': radius,
        'height': height,
        'direction': direction,
    }


def get_sphere_params(
    joint_positions: np.ndarray,
    joint_indices: Dict[str, int],
    part_name: str,
    body_height: float = REFERENCE_BODY_HEIGHT,
) -> Dict:
    """
    Compute sphere parameters from joint positions.

    Args:
        joint_positions: Array of shape (num_joints, 3) with joint positions
        joint_indices: Dict mapping joint names to indices
        part_name: Name of the body part
        body_height: Estimated body height for scaling

    Returns:
        Dict with 'center', 'radius' keys
    """
    config = COLLISION_BODY_PARTS[part_name]
    joints = config['joints']

    if len(joints) != 1:
        raise ValueError(f"Sphere requires exactly 1 joint, got {len(joints)}")

    # Get joint position
    j_idx = joint_indices[joints[0]]
    center = joint_positions[j_idx]

    # Scale radius by body height
    radius = config['radius_scale'] * body_height

    return {
        'center': center,
        'radius': radius,
    }


def get_body_part_params(
    joint_positions: np.ndarray,
    joint_indices: Dict[str, int],
    part_name: str,
    body_height: float = REFERENCE_BODY_HEIGHT,
) -> Dict:
    """
    Compute collision geometry parameters for a body part.

    Args:
        joint_positions: Array of shape (num_joints, 3) with joint positions
        joint_indices: Dict mapping joint names to indices
        part_name: Name of the body part
        body_height: Estimated body height for scaling

    Returns:
        Dict with geometry parameters (type-specific)
    """
    config = COLLISION_BODY_PARTS[part_name]

    if config['type'] == 'capsule':
        params = get_capsule_params(joint_positions, joint_indices, part_name, body_height)
    elif config['type'] == 'sphere':
        params = get_sphere_params(joint_positions, joint_indices, part_name, body_height)
    else:
        raise ValueError(f"Unknown geometry type: {config['type']}")

    params['type'] = config['type']
    params['name'] = part_name

    return params


def are_adjacent(part_a: str, part_b: str) -> bool:
    """Check if two body parts are adjacent (connected)."""
    return frozenset((part_a, part_b)) in ADJACENT_PARTS


def get_contact_tolerance(part_a: str, part_b: str) -> float:
    """Get the contact tolerance for a pair of body parts."""
    pair = (part_a, part_b)
    reverse_pair = (part_b, part_a)

    if pair in SLR_CONTACT_TOLERANCES:
        return SLR_CONTACT_TOLERANCES[pair]
    elif reverse_pair in SLR_CONTACT_TOLERANCES:
        return SLR_CONTACT_TOLERANCES[reverse_pair]
    else:
        return DEFAULT_CONTACT_TOLERANCE


def get_controlling_joints_for_part(part_name: str) -> List[str]:
    """Get the joints that control a body part's position."""
    return CONTROLLING_JOINTS.get(part_name, [])


def _direction_to_quaternion(direction: np.ndarray) -> np.ndarray:
    """
    Convert a direction vector to a quaternion that aligns the Y-axis with that direction.

    PyBullet capsules are oriented along the Y-axis by default.

    Args:
        direction: Unit vector (3,)

    Returns:
        Quaternion (4,) in [x, y, z, w] format
    """
    # Default Y-axis
    y_axis = np.array([0.0, 1.0, 0.0])

    # Handle edge case where direction is parallel to Y-axis
    dot = np.dot(direction, y_axis)
    if abs(dot) > 0.9999:
        if dot > 0:
            # Same direction, identity quaternion
            return np.array([0.0, 0.0, 0.0, 1.0])
        else:
            # Opposite direction, 180° rotation around X or Z
            return np.array([1.0, 0.0, 0.0, 0.0])

    # Compute rotation axis and angle
    axis = np.cross(y_axis, direction)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    # Convert axis-angle to quaternion
    half_angle = angle / 2
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    return np.array([
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        cos_half,
    ])
