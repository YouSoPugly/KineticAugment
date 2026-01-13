"""
Constraint Engine for KineticAugment.

The ConstraintEngine is the central orchestrator for all constraint types.
It can be used to:
- Validate augmented poses against constraints
- Enforce constraints by correcting violations
- Report detailed constraint violation information

Constraint Types:
1. Joint Limits - Anatomical range of motion
2. Velocity Limits - Maximum angular velocities
3. Collision Detection - Self-collision and environment

Example:
    >>> engine = ConstraintEngine(
    ...     joint_limits=True,
    ...     velocity_limits=True,
    ...     collision_detection=False
    ... )
    >>> corrected_params = engine.enforce(augmented_params)
    >>> violations = engine.validate(params, return_violations=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from kinetic_augment.body_model.smplx_wrapper import SMPLXWrapper


@dataclass
class ConstraintViolation:
    """
    Represents a single constraint violation.

    Attributes:
        constraint_type: Type of constraint violated ('joint_limit', 'velocity', etc.)
        joint_name: Name of the joint involved
        axis: Which axis ('x', 'y', 'z') if applicable
        current_value: The violating value
        limit: The limit that was violated (min, max) or max velocity
        severity: 0-1 indicating how severe the violation is
        frame_index: Which frame in a sequence (for temporal constraints)
    """

    constraint_type: str
    joint_name: str
    axis: Optional[str] = None
    current_value: float = 0.0
    limit: Tuple[float, float] = (0.0, 0.0)
    severity: float = 0.0
    frame_index: Optional[int] = None

    def __str__(self) -> str:
        axis_str = f" ({self.axis})" if self.axis else ""
        frame_str = f" @ frame {self.frame_index}" if self.frame_index is not None else ""
        return (
            f"{self.constraint_type}: {self.joint_name}{axis_str} = {self.current_value:.3f} "
            f"(limit: {self.limit}){frame_str}"
        )


class ConstraintEngine:
    """
    Central engine for constraint enforcement and validation.

    The engine can operate in two modes:
    - Validation: Check if poses violate constraints (non-destructive)
    - Enforcement: Correct poses to satisfy constraints (modifies data)

    Attributes:
        joint_limits: Enable joint limit constraints
        velocity_limits: Enable velocity constraints
        collision_detection: Enable collision detection (requires PyBullet)
        strict_mode: If True, raise errors on violations; if False, just report
    """

    def __init__(
        self,
        joint_limits: bool = True,
        velocity_limits: bool = True,
        collision_detection: bool = False,
        strict_mode: bool = False,
        custom_joint_limits: Optional[Dict] = None,
        custom_velocity_limits: Optional[Dict] = None,
        smplx_wrapper: Optional['SMPLXWrapper'] = None,
        collision_config: Optional[Dict] = None,
    ):
        """
        Initialize the constraint engine.

        Args:
            joint_limits: Enable joint angle constraints
            velocity_limits: Enable angular velocity constraints
            collision_detection: Enable self/environment collision
            strict_mode: Raise errors on violations instead of correcting
            custom_joint_limits: Override default joint limits
            custom_velocity_limits: Override default velocity limits
            smplx_wrapper: SMPL-X wrapper for collision detection forward kinematics
            collision_config: Configuration for collision detection:
                - strategy: 'gradient', 'rejection', 'interpolation'
                - use_tiered: bool (default True)
                - collision_pairs: List of (part_a, part_b) tuples
        """
        self.use_joint_limits = joint_limits
        self.use_velocity_limits = velocity_limits
        self.use_collision = collision_detection
        self.strict_mode = strict_mode
        self.smplx_wrapper = smplx_wrapper

        # Initialize constraint handlers
        if self.use_joint_limits:
            from kinetic_augment.constraints.joint_limits import JointLimitConstraint
            self.joint_constraint = JointLimitConstraint(custom_joint_limits)
        else:
            self.joint_constraint = None

        if self.use_velocity_limits:
            from kinetic_augment.constraints.velocity_limits import VelocityConstraint
            self.velocity_constraint = VelocityConstraint(custom_velocity_limits)
        else:
            self.velocity_constraint = None

        # Collision detection (optional, requires PyBullet)
        self.collision_constraint = None
        if self.use_collision:
            try:
                from kinetic_augment.constraints.collision import CollisionConstraint
                collision_config = collision_config or {}
                self.collision_constraint = CollisionConstraint(
                    smplx_wrapper=smplx_wrapper,
                    resolution_strategy=collision_config.get('strategy', 'gradient'),
                    use_tiered=collision_config.get('use_tiered', True),
                    collision_pairs=collision_config.get('collision_pairs'),
                )
                if not self.collision_constraint.is_available:
                    print("Warning: Collision detection requires PyBullet. Disabling.")
                    self.use_collision = False
                    self.collision_constraint = None
            except ImportError:
                print("Warning: Collision detection requires PyBullet. Disabling.")
                self.use_collision = False

    def validate(
        self,
        params: Dict[str, np.ndarray],
        prev_params: Optional[Dict[str, np.ndarray]] = None,
        dt: float = 1/30,
        return_violations: bool = False,
    ) -> bool | Tuple[bool, List[ConstraintViolation]]:
        """
        Validate SMPL-X parameters against all enabled constraints.

        Args:
            params: SMPL-X parameter dictionary
            prev_params: Previous frame parameters (for velocity check)
            dt: Time step in seconds
            return_violations: If True, return list of violations

        Returns:
            If return_violations is False: bool indicating validity
            If return_violations is True: (bool, List[ConstraintViolation])
        """
        violations = []

        # Check joint limits
        if self.joint_constraint is not None:
            joint_violations = self.joint_constraint.check(params)
            violations.extend(joint_violations)

        # Check velocity limits
        if self.velocity_constraint is not None and prev_params is not None:
            velocity_violations = self.velocity_constraint.check(
                params, prev_params, dt
            )
            violations.extend(velocity_violations)

        # Check collisions
        if self.collision_constraint is not None:
            collision_violations = self.collision_constraint.check(params)
            violations.extend(collision_violations)

        is_valid = len(violations) == 0

        if return_violations:
            return is_valid, violations
        return is_valid

    def enforce(
        self,
        params: Dict[str, np.ndarray],
        prev_params: Optional[Dict[str, np.ndarray]] = None,
        dt: float = 1/30,
    ) -> Dict[str, np.ndarray]:
        """
        Enforce constraints by correcting violations.

        Args:
            params: SMPL-X parameter dictionary
            prev_params: Previous frame parameters (for velocity smoothing)
            dt: Time step in seconds

        Returns:
            Corrected parameters satisfying all constraints
        """
        result = {k: v.copy() if isinstance(v, np.ndarray) else v
                  for k, v in params.items()}

        # Enforce joint limits first (anatomical validity)
        if self.joint_constraint is not None:
            result = self.joint_constraint.enforce(result)

        # Enforce velocity limits (temporal smoothness)
        if self.velocity_constraint is not None and prev_params is not None:
            result = self.velocity_constraint.enforce(result, prev_params, dt)

        # Enforce collision constraints last (geometric validity)
        if self.collision_constraint is not None:
            result = self.collision_constraint.enforce(result, prev_params)

        return result

    def enforce_sequence(
        self,
        params_sequence: List[Dict[str, np.ndarray]],
        dt: float = 1/30,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Enforce constraints across an entire sequence.

        Processes frames in order, using previous frame for velocity constraints.

        Args:
            params_sequence: List of SMPL-X parameter dictionaries
            dt: Time step between frames

        Returns:
            List of corrected parameters
        """
        if len(params_sequence) == 0:
            return []

        result = []

        # First frame - no velocity constraint
        first = self.enforce(params_sequence[0])
        result.append(first)

        # Subsequent frames - use previous for velocity
        for i in range(1, len(params_sequence)):
            corrected = self.enforce(
                params_sequence[i],
                prev_params=result[i-1],
                dt=dt
            )
            result.append(corrected)

        return result

    def validate_sequence(
        self,
        params_sequence: List[Dict[str, np.ndarray]],
        dt: float = 1/30,
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """
        Validate an entire sequence.

        Args:
            params_sequence: List of SMPL-X parameter dictionaries
            dt: Time step between frames

        Returns:
            (is_valid, list of all violations with frame indices)
        """
        all_violations = []

        for i, params in enumerate(params_sequence):
            prev_params = params_sequence[i-1] if i > 0 else None

            _, violations = self.validate(
                params, prev_params, dt, return_violations=True
            )

            # Add frame index to violations
            for v in violations:
                v.frame_index = i
            all_violations.extend(violations)

        return len(all_violations) == 0, all_violations

    def get_violation_summary(
        self,
        violations: List[ConstraintViolation]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for constraint violations.

        Args:
            violations: List of ConstraintViolation objects

        Returns:
            Dictionary with violation statistics
        """
        if not violations:
            return {
                'total_violations': 0,
                'by_type': {},
                'by_joint': {},
                'max_severity': 0.0,
            }

        by_type = {}
        by_joint = {}
        severities = []

        for v in violations:
            # Count by type
            by_type[v.constraint_type] = by_type.get(v.constraint_type, 0) + 1

            # Count by joint
            by_joint[v.joint_name] = by_joint.get(v.joint_name, 0) + 1

            # Track severity
            severities.append(v.severity)

        return {
            'total_violations': len(violations),
            'by_type': by_type,
            'by_joint': by_joint,
            'max_severity': max(severities),
            'mean_severity': np.mean(severities),
        }

    def __repr__(self) -> str:
        constraints = []
        if self.use_joint_limits:
            constraints.append("joint_limits")
        if self.use_velocity_limits:
            constraints.append("velocity_limits")
        if self.use_collision:
            constraints.append("collision")

        return f"ConstraintEngine(constraints={constraints})"
