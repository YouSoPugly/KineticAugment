"""
Joint Limit Constraints for KineticAugment.

Enforces anatomical range-of-motion limits based on biomechanics literature.
Joint limits are defined in radians for each axis (x, y, z) of rotation.

The limits are conservative estimates suitable for most adults. Extreme
athletes or children may have different ranges.

Sources:
- Biomechanics and Motor Control of Human Movement (Winter, 2009)
- Human Body Dynamics (Zatsiorsky, 2002)
- Clinical observation data
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from kinetic_augment.constraints.engine import ConstraintViolation
from kinetic_augment.body_model.joint_mapping import (
    SMPLX_BODY_JOINTS,
    SMPLX_JOINT_LIMITS,
    clamp_joint_angles,
)


class JointLimitConstraint:
    """
    Constraint enforcer for joint angle limits.

    Can check SMPL-X body pose against anatomical limits and
    optionally correct violations by clamping.

    Attributes:
        joint_limits: Dictionary of joint limits by name
        tolerance: Small tolerance for floating point comparisons
    """

    def __init__(
        self,
        custom_limits: Optional[Dict] = None,
        tolerance: float = 0.001,
    ):
        """
        Initialize joint limit constraint.

        Args:
            custom_limits: Override default limits for specific joints
            tolerance: Tolerance for limit checking (radians)
        """
        self.joint_limits = dict(SMPLX_JOINT_LIMITS)

        if custom_limits:
            for joint_name, limits in custom_limits.items():
                if joint_name in self.joint_limits:
                    self.joint_limits[joint_name].update(limits)
                else:
                    self.joint_limits[joint_name] = limits

        self.tolerance = tolerance

    def check(
        self,
        params: Dict[str, np.ndarray]
    ) -> List[ConstraintViolation]:
        """
        Check for joint limit violations.

        Args:
            params: SMPL-X parameter dictionary with 'body_pose'

        Returns:
            List of ConstraintViolation objects for each violation
        """
        if 'body_pose' not in params:
            return []

        body_pose = params['body_pose']

        if body_pose.ndim == 1:
            body_pose = body_pose[np.newaxis, ...]

        violations = []

        # Reverse lookup for joint names
        idx_to_name = {v: k for k, v in SMPLX_BODY_JOINTS.items()}

        for joint_idx in range(1, 22):  # Skip pelvis (0)
            joint_name = idx_to_name.get(joint_idx)
            if joint_name is None or joint_name not in self.joint_limits:
                continue

            pose_idx = (joint_idx - 1) * 3
            joint_angles = body_pose[0, pose_idx:pose_idx + 3]
            limits = self.joint_limits[joint_name]

            for i, axis in enumerate(['x', 'y', 'z']):
                if axis not in limits:
                    continue

                min_val, max_val = limits[axis]
                current = joint_angles[i]

                if current < min_val - self.tolerance:
                    severity = (min_val - current) / (max_val - min_val + 0.001)
                    violations.append(ConstraintViolation(
                        constraint_type='joint_limit',
                        joint_name=joint_name,
                        axis=axis,
                        current_value=float(current),
                        limit=(min_val, max_val),
                        severity=min(1.0, severity),
                    ))

                elif current > max_val + self.tolerance:
                    severity = (current - max_val) / (max_val - min_val + 0.001)
                    violations.append(ConstraintViolation(
                        constraint_type='joint_limit',
                        joint_name=joint_name,
                        axis=axis,
                        current_value=float(current),
                        limit=(min_val, max_val),
                        severity=min(1.0, severity),
                    ))

        return violations

    def enforce(
        self,
        params: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Enforce joint limits by clamping violating angles.

        Args:
            params: SMPL-X parameter dictionary

        Returns:
            Parameters with clamped body pose
        """
        if 'body_pose' not in params:
            return params

        result = {k: v.copy() if isinstance(v, np.ndarray) else v
                  for k, v in params.items()}

        body_pose = result['body_pose'].copy()

        if body_pose.ndim == 1:
            body_pose = body_pose[np.newaxis, ...]
            squeeze = True
        else:
            squeeze = False

        batch_size = body_pose.shape[0]
        idx_to_name = {v: k for k, v in SMPLX_BODY_JOINTS.items()}

        for joint_idx in range(1, 22):
            joint_name = idx_to_name.get(joint_idx)
            if joint_name is None:
                continue

            pose_idx = (joint_idx - 1) * 3

            for b in range(batch_size):
                joint_angles = body_pose[b, pose_idx:pose_idx + 3]
                clamped = clamp_joint_angles(
                    joint_name, joint_angles, self.joint_limits
                )
                body_pose[b, pose_idx:pose_idx + 3] = clamped

        if squeeze:
            body_pose = body_pose[0]

        result['body_pose'] = body_pose
        return result


def enforce_joint_limits(
    params: Dict[str, np.ndarray],
    custom_limits: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to enforce joint limits on SMPL-X parameters.

    Args:
        params: SMPL-X parameter dictionary
        custom_limits: Optional custom joint limits

    Returns:
        Parameters with enforced joint limits
    """
    constraint = JointLimitConstraint(custom_limits)
    return constraint.enforce(params)


# Extended joint limits for specific use cases

RELAXED_JOINT_LIMITS = {
    # Allow slightly more range for artistic/dance movements
    'left_shoulder': {
        'x': (-0.7, 3.5),   # Extended flexion
        'y': (-1.75, 0.7),
        'z': (-1.75, 1.75),
    },
    'right_shoulder': {
        'x': (-0.7, 3.5),
        'y': (-0.7, 1.75),
        'z': (-1.75, 1.75),
    },
}

STRICT_JOINT_LIMITS = {
    # More conservative limits for everyday motion
    'left_shoulder': {
        'x': (-0.35, 2.5),
        'y': (-1.2, 0.35),
        'z': (-1.2, 1.2),
    },
    'right_shoulder': {
        'x': (-0.35, 2.5),
        'y': (-0.35, 1.2),
        'z': (-1.2, 1.2),
    },
    'left_elbow': {
        'x': (0, 2.35),
        'y': (-0.1, 0.1),
        'z': (-1.2, 1.2),
    },
    'right_elbow': {
        'x': (0, 2.35),
        'y': (-0.1, 0.1),
        'z': (-1.2, 1.2),
    },
}
