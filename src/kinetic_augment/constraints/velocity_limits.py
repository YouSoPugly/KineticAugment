"""
Velocity Constraints for KineticAugment.

Enforces maximum angular velocities to ensure temporal coherence.
Prevents physically impossible sudden movements (jerk reduction).

Velocity limits are defined in radians per second. These limits
are based on:
- Human motor control literature
- Motion capture studies
- Sign language movement analysis

For Sign Language Recognition, hand/arm velocities are typically
higher than full-body motion, so we use separate limits.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from kinetic_augment.constraints.engine import ConstraintViolation
from kinetic_augment.body_model.joint_mapping import SMPLX_BODY_JOINTS


# Default maximum angular velocities (radians/second)
DEFAULT_VELOCITY_LIMITS = {
    # Spine - slow, controlled movement
    'spine1': 3.0,
    'spine2': 3.0,
    'spine3': 3.0,
    'neck': 5.0,
    'head': 6.0,

    # Shoulders - medium speed
    'left_shoulder': 8.0,
    'right_shoulder': 8.0,
    'left_collar': 4.0,
    'right_collar': 4.0,

    # Elbows - fast
    'left_elbow': 12.0,
    'right_elbow': 12.0,

    # Wrists - very fast (for signing)
    'left_wrist': 15.0,
    'right_wrist': 15.0,

    # Hips - medium
    'left_hip': 6.0,
    'right_hip': 6.0,

    # Knees - medium-fast
    'left_knee': 10.0,
    'right_knee': 10.0,

    # Ankles - medium
    'left_ankle': 8.0,
    'right_ankle': 8.0,
    'left_foot': 10.0,
    'right_foot': 10.0,
}

# Higher limits for sign language (hands move faster)
SIGN_LANGUAGE_VELOCITY_LIMITS = {
    **DEFAULT_VELOCITY_LIMITS,
    'left_shoulder': 12.0,
    'right_shoulder': 12.0,
    'left_elbow': 18.0,
    'right_elbow': 18.0,
    'left_wrist': 25.0,
    'right_wrist': 25.0,
}


class VelocityConstraint:
    """
    Constraint enforcer for angular velocity limits.

    Ensures temporal coherence by limiting how fast joints can move
    between frames. Uses exponential smoothing for enforcement.

    Attributes:
        velocity_limits: Maximum angular velocity per joint (rad/s)
        smoothing_factor: For enforcement, how much to blend toward limit
    """

    def __init__(
        self,
        custom_limits: Optional[Dict[str, float]] = None,
        smoothing_factor: float = 0.7,
        use_sign_language_limits: bool = False,
    ):
        """
        Initialize velocity constraint.

        Args:
            custom_limits: Override default limits for specific joints
            smoothing_factor: Blending factor for enforcement (0-1)
            use_sign_language_limits: Use higher limits for signing
        """
        if use_sign_language_limits:
            self.velocity_limits = dict(SIGN_LANGUAGE_VELOCITY_LIMITS)
        else:
            self.velocity_limits = dict(DEFAULT_VELOCITY_LIMITS)

        if custom_limits:
            self.velocity_limits.update(custom_limits)

        self.smoothing_factor = smoothing_factor

    def check(
        self,
        params: Dict[str, np.ndarray],
        prev_params: Dict[str, np.ndarray],
        dt: float = 1/30,
    ) -> List[ConstraintViolation]:
        """
        Check for velocity limit violations.

        Args:
            params: Current SMPL-X parameters
            prev_params: Previous frame parameters
            dt: Time step in seconds

        Returns:
            List of velocity violations
        """
        if 'body_pose' not in params or 'body_pose' not in prev_params:
            return []

        current_pose = params['body_pose']
        prev_pose = prev_params['body_pose']

        if current_pose.ndim == 1:
            current_pose = current_pose[np.newaxis, ...]
        if prev_pose.ndim == 1:
            prev_pose = prev_pose[np.newaxis, ...]

        violations = []
        idx_to_name = {v: k for k, v in SMPLX_BODY_JOINTS.items()}

        for joint_idx in range(1, 22):
            joint_name = idx_to_name.get(joint_idx)
            if joint_name is None:
                continue

            max_velocity = self.velocity_limits.get(joint_name, 10.0)
            pose_idx = (joint_idx - 1) * 3

            current_angles = current_pose[0, pose_idx:pose_idx + 3]
            prev_angles = prev_pose[0, pose_idx:pose_idx + 3]

            # Compute angular velocity magnitude
            delta = current_angles - prev_angles
            velocity = np.linalg.norm(delta) / dt

            if velocity > max_velocity:
                severity = (velocity - max_velocity) / max_velocity
                violations.append(ConstraintViolation(
                    constraint_type='velocity',
                    joint_name=joint_name,
                    current_value=float(velocity),
                    limit=(0, max_velocity),
                    severity=min(1.0, severity),
                ))

        return violations

    def enforce(
        self,
        params: Dict[str, np.ndarray],
        prev_params: Dict[str, np.ndarray],
        dt: float = 1/30,
    ) -> Dict[str, np.ndarray]:
        """
        Enforce velocity limits by smoothing motion.

        Uses exponential smoothing to blend current pose toward
        a velocity-compliant pose.

        Args:
            params: Current SMPL-X parameters
            prev_params: Previous frame parameters
            dt: Time step in seconds

        Returns:
            Parameters with enforced velocity limits
        """
        if 'body_pose' not in params or 'body_pose' not in prev_params:
            return params

        result = {k: v.copy() if isinstance(v, np.ndarray) else v
                  for k, v in params.items()}

        current_pose = result['body_pose'].copy()
        prev_pose = prev_params['body_pose']

        if current_pose.ndim == 1:
            current_pose = current_pose[np.newaxis, ...]
            squeeze = True
        else:
            squeeze = False

        if prev_pose.ndim == 1:
            prev_pose = prev_pose[np.newaxis, ...]

        batch_size = current_pose.shape[0]
        idx_to_name = {v: k for k, v in SMPLX_BODY_JOINTS.items()}

        for joint_idx in range(1, 22):
            joint_name = idx_to_name.get(joint_idx)
            if joint_name is None:
                continue

            max_velocity = self.velocity_limits.get(joint_name, 10.0)
            max_delta = max_velocity * dt

            pose_idx = (joint_idx - 1) * 3

            for b in range(batch_size):
                current_angles = current_pose[b, pose_idx:pose_idx + 3]
                prev_angles = prev_pose[b, pose_idx:pose_idx + 3]

                delta = current_angles - prev_angles
                delta_magnitude = np.linalg.norm(delta)

                if delta_magnitude > max_delta:
                    # Scale delta to max velocity
                    scale = max_delta / delta_magnitude
                    limited_delta = delta * scale

                    # Blend between original and limited
                    blended_delta = (
                        self.smoothing_factor * limited_delta +
                        (1 - self.smoothing_factor) * delta
                    )

                    current_pose[b, pose_idx:pose_idx + 3] = prev_angles + blended_delta

        if squeeze:
            current_pose = current_pose[0]

        result['body_pose'] = current_pose
        return result


def enforce_velocity_limits(
    params: Dict[str, np.ndarray],
    prev_params: Dict[str, np.ndarray],
    dt: float = 1/30,
    custom_limits: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to enforce velocity limits.

    Args:
        params: Current SMPL-X parameters
        prev_params: Previous frame parameters
        dt: Time step in seconds
        custom_limits: Optional custom velocity limits

    Returns:
        Parameters with enforced velocity limits
    """
    constraint = VelocityConstraint(custom_limits)
    return constraint.enforce(params, prev_params, dt)


def compute_sequence_velocities(
    params_sequence: List[Dict[str, np.ndarray]],
    dt: float = 1/30,
) -> Dict[str, np.ndarray]:
    """
    Compute angular velocities for each joint across a sequence.

    Useful for analysis and visualization.

    Args:
        params_sequence: List of SMPL-X parameter dictionaries
        dt: Time step between frames

    Returns:
        Dictionary mapping joint names to velocity arrays (num_frames-1,)
    """
    if len(params_sequence) < 2:
        return {}

    idx_to_name = {v: k for k, v in SMPLX_BODY_JOINTS.items()}
    velocities = {name: [] for name in idx_to_name.values()}

    for i in range(1, len(params_sequence)):
        current_pose = params_sequence[i]['body_pose']
        prev_pose = params_sequence[i-1]['body_pose']

        if current_pose.ndim == 2:
            current_pose = current_pose[0]
        if prev_pose.ndim == 2:
            prev_pose = prev_pose[0]

        for joint_idx in range(1, 22):
            joint_name = idx_to_name.get(joint_idx)
            if joint_name is None:
                continue

            pose_idx = (joint_idx - 1) * 3

            delta = current_pose[pose_idx:pose_idx + 3] - prev_pose[pose_idx:pose_idx + 3]
            velocity = np.linalg.norm(delta) / dt

            velocities[joint_name].append(velocity)

    return {k: np.array(v) for k, v in velocities.items()}
