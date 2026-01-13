"""
Collision Constraint for KineticAugment.

Provides self-collision detection and resolution as a constraint
that integrates with the ConstraintEngine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from kinetic_augment.constraints.engine import ConstraintViolation

if TYPE_CHECKING:
    from kinetic_augment.body_model.smplx_wrapper import SMPLXWrapper

# Try to import PyBullet-dependent modules
try:
    from kinetic_augment.constraints.collision.collision_checker import (
        CollisionChecker,
        TieredCollisionChecker,
        ContactPoint,
        PYBULLET_AVAILABLE,
    )
    from kinetic_augment.constraints.collision.resolution import (
        GradientResolver,
        RejectionResolver,
        InterpolationResolver,
        create_resolver,
    )
    from kinetic_augment.constraints.collision.body_parts import (
        SLR_COLLISION_PAIRS,
        FULL_COLLISION_PAIRS,
    )
except ImportError:
    PYBULLET_AVAILABLE = False
    CollisionChecker = None
    TieredCollisionChecker = None


@dataclass
class CollisionViolation:
    """
    Collision-specific violation information.

    Similar to ConstraintViolation but specialized for collisions.
    """
    constraint_type: str = 'collision'
    joint_name: str = ''  # Combined name like 'left_hand_torso'
    body_part_a: str = ''
    body_part_b: str = ''
    penetration_depth: float = 0.0
    current_value: float = 0.0  # Same as penetration_depth for compatibility
    limit: Tuple[float, float] = (0.0, 0.0)  # No penetration allowed
    severity: float = 0.0
    frame_index: Optional[int] = None
    contact_point: Optional[np.ndarray] = None
    contact_normal: Optional[np.ndarray] = None

    def __str__(self) -> str:
        frame_str = f" @ frame {self.frame_index}" if self.frame_index is not None else ""
        return (
            f"collision: {self.body_part_a} <-> {self.body_part_b}, "
            f"depth={self.penetration_depth:.4f}m{frame_str}"
        )


class CollisionConstraint:
    """
    Constraint class for self-collision detection and resolution.

    Follows the same interface as JointLimitConstraint and VelocityConstraint:
    - check(params) -> List[ConstraintViolation]: Detect violations
    - enforce(params) -> Dict: Correct violations

    Requires PyBullet to be installed. If not available, check() returns
    empty list and enforce() returns params unchanged.

    Example:
        >>> constraint = CollisionConstraint(smplx_wrapper)
        >>> violations = constraint.check(params)
        >>> corrected = constraint.enforce(params)
    """

    def __init__(
        self,
        smplx_wrapper: Optional['SMPLXWrapper'] = None,
        resolution_strategy: str = 'gradient',
        use_tiered: bool = True,
        collision_pairs: Optional[List[Tuple[str, str]]] = None,
        body_height: float = 1.7,
    ):
        """
        Initialize collision constraint.

        Args:
            smplx_wrapper: SMPL-X wrapper for forward kinematics.
                          Required for iterative resolution.
            resolution_strategy: How to resolve collisions:
                - 'gradient': Push parts apart (recommended)
                - 'rejection': Fall back to previous valid pose
                - 'interpolation': Binary search for valid blend
            use_tiered: If True, use tiered checker (capsule + convex)
            collision_pairs: Which body part pairs to check.
                           Defaults to SLR_COLLISION_PAIRS.
            body_height: Body height for scaling collision geometry
        """
        self.smplx_wrapper = smplx_wrapper
        self.resolution_strategy = resolution_strategy
        self.use_tiered = use_tiered
        self.collision_pairs = collision_pairs
        self.body_height = body_height

        # Lazy initialization
        self._checker: Optional[CollisionChecker] = None
        self._resolver = None
        self._is_available = PYBULLET_AVAILABLE

    @property
    def is_available(self) -> bool:
        """Check if collision detection is available (PyBullet installed)."""
        return self._is_available

    def _ensure_checker(self) -> bool:
        """Initialize collision checker if needed. Returns True if successful."""
        if not self._is_available:
            return False

        if self._checker is None:
            try:
                if self.use_tiered:
                    self._checker = TieredCollisionChecker(
                        collision_pairs=self.collision_pairs,
                        body_height=self.body_height,
                    )
                else:
                    self._checker = CollisionChecker(
                        collision_pairs=self.collision_pairs,
                        body_height=self.body_height,
                    )
                self._checker.setup()
            except Exception as e:
                print(f"Warning: Failed to initialize collision checker: {e}")
                self._is_available = False
                return False

        return True

    def _ensure_resolver(self) -> None:
        """Initialize resolver if needed."""
        if self._resolver is None and self._is_available:
            self._resolver = create_resolver(self.resolution_strategy)

    def check(
        self,
        params: Dict[str, np.ndarray],
        smplx_output: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[CollisionViolation]:
        """
        Check for self-collisions.

        Args:
            params: SMPL-X parameters dict
            smplx_output: Pre-computed SMPL-X forward output with 'joints'.
                         If None and smplx_wrapper available, computes it.

        Returns:
            List of CollisionViolation objects for detected collisions
        """
        if not self._ensure_checker():
            return []

        # Get joint positions
        joints = self._get_joints(params, smplx_output)
        if joints is None:
            return []

        # Update collision geometry
        self._checker.update_from_joints(joints)

        # Check for collisions
        contacts = self._checker.get_contact_points()

        # Convert to violations
        violations = []
        for contact in contacts:
            violation = CollisionViolation(
                constraint_type='collision',
                joint_name=f"{contact.body_part_a}_{contact.body_part_b}",
                body_part_a=contact.body_part_a,
                body_part_b=contact.body_part_b,
                penetration_depth=contact.depth,
                current_value=contact.depth,
                limit=(0.0, 0.0),  # No penetration allowed
                severity=min(contact.depth * 10, 1.0),  # Scale depth to severity
                contact_point=contact.position,
                contact_normal=contact.normal,
            )
            violations.append(violation)

        return violations

    def enforce(
        self,
        params: Dict[str, np.ndarray],
        prev_params: Optional[Dict[str, np.ndarray]] = None,
        smplx_output: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Resolve collisions by adjusting parameters.

        Args:
            params: SMPL-X parameters dict
            prev_params: Previous frame parameters (for rejection/interpolation)
            smplx_output: Pre-computed SMPL-X forward output

        Returns:
            Corrected parameters with collisions resolved (hopefully)
        """
        if not self._ensure_checker():
            return params

        self._ensure_resolver()

        # Get joint positions and check for collisions
        joints = self._get_joints(params, smplx_output)
        if joints is None:
            return params

        self._checker.update_from_joints(joints)
        contacts = self._checker.get_contact_points()

        if not contacts:
            return params

        # Resolve collisions
        if self.resolution_strategy == 'rejection':
            resolved = self._resolver.resolve(
                params, contacts,
                prev_params=prev_params,
            )
        elif self.resolution_strategy == 'interpolation':
            resolved = self._resolver.resolve(
                params, contacts,
                smplx_wrapper=self.smplx_wrapper,
                collision_checker=self._checker,
                prev_params=prev_params,
            )
        else:  # gradient
            resolved = self._resolver.resolve(
                params, contacts,
                smplx_wrapper=self.smplx_wrapper,
                collision_checker=self._checker,
            )

        return resolved

    def _get_joints(
        self,
        params: Dict[str, np.ndarray],
        smplx_output: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        """
        Get joint positions from params or smplx_output.

        Args:
            params: SMPL-X parameters
            smplx_output: Pre-computed forward output

        Returns:
            Joint positions array (num_joints, 3) or None
        """
        if smplx_output is not None and 'joints' in smplx_output:
            return smplx_output['joints']

        if self.smplx_wrapper is not None:
            output = self.smplx_wrapper.forward(**params, return_vertices=False)
            return output['joints']

        # Can't get joints without wrapper or pre-computed output
        return None

    def cleanup(self) -> None:
        """Release resources."""
        if self._checker is not None:
            self._checker.cleanup()
            self._checker = None

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()

    def __repr__(self) -> str:
        status = "available" if self._is_available else "unavailable"
        return (
            f"CollisionConstraint(strategy={self.resolution_strategy}, "
            f"tiered={self.use_tiered}, status={status})"
        )


def check_pybullet_available() -> bool:
    """Check if PyBullet is available for collision detection."""
    return PYBULLET_AVAILABLE
