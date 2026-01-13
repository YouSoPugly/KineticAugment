"""
Joint Mapping between SMPL-X and MediaPipe.

This module defines the correspondence between SMPL-X body model joints
and MediaPipe Holistic landmarks, enabling conversion between the two formats.

SMPL-X Joint Hierarchy (55 joints for body + hands):
    - Body: 22 joints (pelvis, spine, head, arms, legs)
    - Jaw: 1 joint
    - Eyes: 2 joints
    - Left Hand: 15 joints (MANO)
    - Right Hand: 15 joints (MANO)

MediaPipe Pose Landmarks (33 points):
    - Face: nose, eyes (6), ears (2), mouth (2)
    - Body: shoulders, elbows, wrists, hips, knees, ankles
    - Hands: thumb, index, pinky (basic landmarks only)
    - Feet: heels, foot indices

Note: MediaPipe hand landmarks (21 per hand) are separate from pose landmarks.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

# =============================================================================
# SMPL-X Joint Definitions
# =============================================================================

# SMPL-X body joint indices (22 joints)
# Reference: https://github.com/vchoutas/smplx
SMPLX_BODY_JOINTS = {
    'pelvis': 0,
    'left_hip': 1,
    'right_hip': 2,
    'spine1': 3,
    'left_knee': 4,
    'right_knee': 5,
    'spine2': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'spine3': 9,
    'left_foot': 10,
    'right_foot': 11,
    'neck': 12,
    'left_collar': 13,
    'right_collar': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
}

# Additional SMPL-X joints (jaw and eyes)
SMPLX_EXTRA_JOINTS = {
    'jaw': 22,
    'left_eye': 23,
    'right_eye': 24,
}

# SMPL-X hand joint indices (15 joints per hand, MANO format)
# These are offsets from the hand start index
SMPLX_HAND_JOINTS = {
    'index1': 0,
    'index2': 1,
    'index3': 2,
    'middle1': 3,
    'middle2': 4,
    'middle3': 5,
    'pinky1': 6,
    'pinky2': 7,
    'pinky3': 8,
    'ring1': 9,
    'ring2': 10,
    'ring3': 11,
    'thumb1': 12,
    'thumb2': 13,
    'thumb3': 14,
}

# Hand joint start indices in full SMPL-X joint array
SMPLX_LEFT_HAND_START = 25
SMPLX_RIGHT_HAND_START = 40

# Total SMPL-X joints
SMPLX_NUM_BODY_JOINTS = 22
SMPLX_NUM_HAND_JOINTS = 15
SMPLX_NUM_TOTAL_JOINTS = 55  # 22 body + 1 jaw + 2 eyes + 15 left hand + 15 right hand

# =============================================================================
# MediaPipe Pose Landmark Definitions
# =============================================================================

MEDIAPIPE_POSE_LANDMARKS = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32,
}

# MediaPipe Hand Landmarks (21 per hand)
MEDIAPIPE_HAND_LANDMARKS = {
    'wrist': 0,
    'thumb_cmc': 1,
    'thumb_mcp': 2,
    'thumb_ip': 3,
    'thumb_tip': 4,
    'index_mcp': 5,
    'index_pip': 6,
    'index_dip': 7,
    'index_tip': 8,
    'middle_mcp': 9,
    'middle_pip': 10,
    'middle_dip': 11,
    'middle_tip': 12,
    'ring_mcp': 13,
    'ring_pip': 14,
    'ring_dip': 15,
    'ring_tip': 16,
    'pinky_mcp': 17,
    'pinky_pip': 18,
    'pinky_dip': 19,
    'pinky_tip': 20,
}

# =============================================================================
# Joint Correspondence Mapping
# =============================================================================

# SMPL-X to MediaPipe Pose mapping
# Format: {smplx_joint_name: mediapipe_landmark_name}
# Note: Not all joints have direct correspondences
SMPLX_TO_MEDIAPIPE_POSE: Dict[str, str] = {
    # Hips
    'left_hip': 'left_hip',
    'right_hip': 'right_hip',

    # Legs
    'left_knee': 'left_knee',
    'right_knee': 'right_knee',
    'left_ankle': 'left_ankle',
    'right_ankle': 'right_ankle',
    'left_foot': 'left_foot_index',  # Approximate
    'right_foot': 'right_foot_index',  # Approximate

    # Arms
    'left_shoulder': 'left_shoulder',
    'right_shoulder': 'right_shoulder',
    'left_elbow': 'left_elbow',
    'right_elbow': 'right_elbow',
    'left_wrist': 'left_wrist',
    'right_wrist': 'right_wrist',

    # Head area
    'head': 'nose',  # Approximate - head center vs nose
    'left_eye': 'left_eye',
    'right_eye': 'right_eye',
}

# Reverse mapping: MediaPipe to SMPL-X
MEDIAPIPE_TO_SMPLX_POSE: Dict[str, str] = {v: k for k, v in SMPLX_TO_MEDIAPIPE_POSE.items()}

# SMPL-X hand joints to MediaPipe hand landmarks mapping
# Note: SMPL-X uses MANO which has different joint organization
SMPLX_TO_MEDIAPIPE_HAND: Dict[str, str] = {
    'thumb1': 'thumb_cmc',
    'thumb2': 'thumb_mcp',
    'thumb3': 'thumb_ip',
    # thumb_tip is not a SMPL-X joint (it's derived)

    'index1': 'index_mcp',
    'index2': 'index_pip',
    'index3': 'index_dip',

    'middle1': 'middle_mcp',
    'middle2': 'middle_pip',
    'middle3': 'middle_dip',

    'ring1': 'ring_mcp',
    'ring2': 'ring_pip',
    'ring3': 'ring_dip',

    'pinky1': 'pinky_mcp',
    'pinky2': 'pinky_pip',
    'pinky3': 'pinky_dip',
}

# =============================================================================
# Named Lists for Easy Access
# =============================================================================

SMPLX_JOINT_NAMES: List[str] = list(SMPLX_BODY_JOINTS.keys())
MEDIAPIPE_POSE_NAMES: List[str] = list(MEDIAPIPE_POSE_LANDMARKS.keys())
MEDIAPIPE_HAND_NAMES: List[str] = list(MEDIAPIPE_HAND_LANDMARKS.keys())

# =============================================================================
# Joint Limit Definitions (in radians)
# =============================================================================

# Anatomical joint limits based on biomechanics literature
# Format: {joint_name: {axis: (min, max)}}
# Axes follow SMPL-X convention: rotation around x, y, z local axes
SMPLX_JOINT_LIMITS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # Spine - limited mobility
    'spine1': {
        'x': (-0.52, 0.70),   # -30° to 40° (flexion/extension)
        'y': (-0.52, 0.52),   # -30° to 30° (lateral bend)
        'z': (-0.52, 0.52),   # -30° to 30° (axial rotation)
    },
    'spine2': {
        'x': (-0.35, 0.52),
        'y': (-0.35, 0.35),
        'z': (-0.35, 0.35),
    },
    'spine3': {
        'x': (-0.35, 0.52),
        'y': (-0.35, 0.35),
        'z': (-0.35, 0.35),
    },

    # Neck
    'neck': {
        'x': (-0.87, 1.05),   # -50° to 60° (nod)
        'y': (-1.22, 1.22),   # -70° to 70° (turn)
        'z': (-0.70, 0.70),   # -40° to 40° (tilt)
    },

    # Head (relative to neck)
    'head': {
        'x': (-0.35, 0.52),
        'y': (-0.52, 0.52),
        'z': (-0.35, 0.35),
    },

    # Shoulders
    'left_shoulder': {
        'x': (-0.52, 3.14),   # -30° to 180° (flexion)
        'y': (-1.57, 0.52),   # -90° to 30° (abduction plane)
        'z': (-1.57, 1.57),   # -90° to 90° (rotation)
    },
    'right_shoulder': {
        'x': (-0.52, 3.14),
        'y': (-0.52, 1.57),   # Mirrored
        'z': (-1.57, 1.57),
    },

    # Elbows - hinge joints
    'left_elbow': {
        'x': (0, 2.62),       # 0° to 150° (flexion only)
        'y': (-0.17, 0.17),   # Very limited
        'z': (-1.57, 1.57),   # Pronation/supination
    },
    'right_elbow': {
        'x': (0, 2.62),
        'y': (-0.17, 0.17),
        'z': (-1.57, 1.57),
    },

    # Wrists
    'left_wrist': {
        'x': (-1.22, 1.22),   # -70° to 70° (flexion/extension)
        'y': (-0.52, 0.35),   # -30° to 20° (ulnar/radial deviation)
        'z': (-0.17, 0.17),   # Very limited rotation
    },
    'right_wrist': {
        'x': (-1.22, 1.22),
        'y': (-0.35, 0.52),   # Mirrored
        'z': (-0.17, 0.17),
    },

    # Hips
    'left_hip': {
        'x': (-0.52, 2.09),   # -30° to 120° (flexion)
        'y': (-0.79, 0.70),   # -45° to 40° (abduction/adduction)
        'z': (-0.79, 0.79),   # -45° to 45° (rotation)
    },
    'right_hip': {
        'x': (-0.52, 2.09),
        'y': (-0.70, 0.79),   # Mirrored
        'z': (-0.79, 0.79),
    },

    # Knees - hinge joints
    'left_knee': {
        'x': (0, 2.62),       # 0° to 150° (flexion only)
        'y': (-0.09, 0.09),   # Very limited
        'z': (-0.09, 0.09),   # Very limited
    },
    'right_knee': {
        'x': (0, 2.62),
        'y': (-0.09, 0.09),
        'z': (-0.09, 0.09),
    },

    # Ankles
    'left_ankle': {
        'x': (-0.70, 0.79),   # -40° to 45° (dorsi/plantar flexion)
        'y': (-0.52, 0.35),   # -30° to 20° (inversion/eversion)
        'z': (-0.35, 0.35),   # Limited rotation
    },
    'right_ankle': {
        'x': (-0.70, 0.79),
        'y': (-0.35, 0.52),   # Mirrored
        'z': (-0.35, 0.35),
    },
}

# Finger joint limits (simplified - same for all fingers except thumb)
FINGER_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    'mcp_flexion': (0, 1.57),      # 0° to 90°
    'mcp_abduction': (-0.35, 0.35), # -20° to 20°
    'pip_flexion': (0, 1.75),      # 0° to 100°
    'dip_flexion': (0, 1.40),      # 0° to 80°
}

THUMB_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    'cmc_flexion': (-0.52, 0.87),   # -30° to 50°
    'cmc_abduction': (-0.70, 0.70), # -40° to 40°
    'mcp_flexion': (0, 1.22),       # 0° to 70°
    'ip_flexion': (0, 1.40),        # 0° to 80°
}

# =============================================================================
# Utility Functions
# =============================================================================

def get_corresponding_joints(source_format: str = 'mediapipe') -> Dict[int, int]:
    """
    Get index-based mapping between MediaPipe and SMPL-X joints.

    Args:
        source_format: 'mediapipe' or 'smplx'

    Returns:
        Dictionary mapping source indices to target indices
    """
    if source_format == 'mediapipe':
        mapping = {}
        for mp_name, smplx_name in MEDIAPIPE_TO_SMPLX_POSE.items():
            if mp_name in MEDIAPIPE_POSE_LANDMARKS and smplx_name in SMPLX_BODY_JOINTS:
                mp_idx = MEDIAPIPE_POSE_LANDMARKS[mp_name]
                smplx_idx = SMPLX_BODY_JOINTS[smplx_name]
                mapping[mp_idx] = smplx_idx
        return mapping
    elif source_format == 'smplx':
        mapping = {}
        for smplx_name, mp_name in SMPLX_TO_MEDIAPIPE_POSE.items():
            if smplx_name in SMPLX_BODY_JOINTS and mp_name in MEDIAPIPE_POSE_LANDMARKS:
                smplx_idx = SMPLX_BODY_JOINTS[smplx_name]
                mp_idx = MEDIAPIPE_POSE_LANDMARKS[mp_name]
                mapping[smplx_idx] = mp_idx
        return mapping
    else:
        raise ValueError(f"Unknown source format: {source_format}")


def get_smplx_joint_index(joint_name: str) -> int:
    """Get the index of a SMPL-X joint by name."""
    if joint_name in SMPLX_BODY_JOINTS:
        return SMPLX_BODY_JOINTS[joint_name]
    elif joint_name in SMPLX_EXTRA_JOINTS:
        return SMPLX_EXTRA_JOINTS[joint_name]
    else:
        raise ValueError(f"Unknown SMPL-X joint: {joint_name}")


def get_mediapipe_landmark_index(landmark_name: str) -> int:
    """Get the index of a MediaPipe pose landmark by name."""
    if landmark_name in MEDIAPIPE_POSE_LANDMARKS:
        return MEDIAPIPE_POSE_LANDMARKS[landmark_name]
    else:
        raise ValueError(f"Unknown MediaPipe landmark: {landmark_name}")


def clamp_joint_angles(
    joint_name: str,
    angles: np.ndarray,
    joint_limits: Optional[Dict] = None
) -> np.ndarray:
    """
    Clamp joint angles to anatomical limits.

    Args:
        joint_name: Name of the joint
        angles: Array of shape (3,) with x, y, z rotation angles in radians
        joint_limits: Optional custom joint limits dict

    Returns:
        Clamped angles
    """
    limits = joint_limits or SMPLX_JOINT_LIMITS

    if joint_name not in limits:
        return angles  # No limits defined for this joint

    clamped = angles.copy()
    joint_limit = limits[joint_name]

    for i, axis in enumerate(['x', 'y', 'z']):
        if axis in joint_limit:
            min_val, max_val = joint_limit[axis]
            clamped[i] = np.clip(clamped[i], min_val, max_val)

    return clamped


def get_kinematic_chain(end_joint: str, start_joint: str = 'pelvis') -> List[str]:
    """
    Get the kinematic chain from start to end joint.

    Args:
        end_joint: Target joint name
        start_joint: Root joint name (default: pelvis)

    Returns:
        List of joint names in the chain
    """
    # Define parent relationships
    PARENT_JOINTS = {
        'pelvis': None,
        'spine1': 'pelvis',
        'spine2': 'spine1',
        'spine3': 'spine2',
        'neck': 'spine3',
        'head': 'neck',
        'left_collar': 'spine3',
        'right_collar': 'spine3',
        'left_shoulder': 'left_collar',
        'right_shoulder': 'right_collar',
        'left_elbow': 'left_shoulder',
        'right_elbow': 'right_shoulder',
        'left_wrist': 'left_elbow',
        'right_wrist': 'right_elbow',
        'left_hip': 'pelvis',
        'right_hip': 'pelvis',
        'left_knee': 'left_hip',
        'right_knee': 'right_hip',
        'left_ankle': 'left_knee',
        'right_ankle': 'right_knee',
        'left_foot': 'left_ankle',
        'right_foot': 'right_ankle',
    }

    # Build chain from end to start
    chain = []
    current = end_joint

    while current is not None and current != start_joint:
        chain.append(current)
        current = PARENT_JOINTS.get(current)

    if current == start_joint:
        chain.append(start_joint)

    # Reverse to get start-to-end order
    return chain[::-1]


# =============================================================================
# Limb Definitions for Inter-Limb Coordination
# =============================================================================

# Define limb groups for coordinated augmentation
LIMB_GROUPS = {
    'left_arm': ['left_shoulder', 'left_elbow', 'left_wrist'],
    'right_arm': ['right_shoulder', 'right_elbow', 'right_wrist'],
    'left_leg': ['left_hip', 'left_knee', 'left_ankle'],
    'right_leg': ['right_hip', 'right_knee', 'right_ankle'],
    'spine': ['spine1', 'spine2', 'spine3', 'neck', 'head'],
}

# Symmetric limb pairs
SYMMETRIC_LIMBS = [
    ('left_arm', 'right_arm'),
    ('left_leg', 'right_leg'),
]

# Anti-phase limb pairs (for walking/running)
ANTIPHASE_LIMBS = [
    ('left_arm', 'right_leg'),
    ('right_arm', 'left_leg'),
]
