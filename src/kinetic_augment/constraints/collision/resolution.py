"""
Collision Resolution Strategies.

Implements methods to resolve detected self-collisions by adjusting
SMPL-X body pose parameters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np

from kinetic_augment.body_model.joint_mapping import (
    SMPLX_BODY_JOINTS,
    clamp_joint_angles,
    SMPLX_JOINT_LIMITS,
)
from kinetic_augment.constraints.collision.body_parts import (
    CONTROLLING_JOINTS,
    get_controlling_joints_for_part,
)
from kinetic_augment.constraints.collision.collision_checker import ContactPoint

if TYPE_CHECKING:
    from kinetic_augment.body_model.smplx_wrapper import SMPLXWrapper
    from kinetic_augment.constraints.collision.collision_checker import CollisionChecker


class CollisionResolver(ABC):
    """Abstract base class for collision resolution strategies."""

    @abstractmethod
    def resolve(
        self,
        params: Dict[str, np.ndarray],
        contacts: List[ContactPoint],
        smplx_wrapper: Optional['SMPLXWrapper'] = None,
        collision_checker: Optional['CollisionChecker'] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Resolve collisions and return corrected parameters.

        Args:
            params: SMPL-X parameters dict with 'body_pose', etc.
            contacts: List of ContactPoint objects describing collisions
            smplx_wrapper: Optional SMPL-X wrapper for forward kinematics
            collision_checker: Optional collision checker for iterative resolution

        Returns:
            Corrected SMPL-X parameters dict
        """
        pass


class GradientResolver(CollisionResolver):
    """
    Gradient-based collision resolution.

    Pushes colliding body parts apart by adjusting joint angles.
    Uses contact normals to determine push direction and applies
    small joint rotations to the controlling joints.
    """

    def __init__(
        self,
        step_size: float = 0.05,
        max_iterations: int = 10,
        convergence_threshold: float = 0.001,
        respect_limits: bool = True,
    ):
        """
        Initialize gradient resolver.

        Args:
            step_size: Base rotation step in radians
            max_iterations: Maximum resolution iterations
            convergence_threshold: Stop when max penetration below this (meters)
            respect_limits: Whether to clamp to anatomical joint limits
        """
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.respect_limits = respect_limits

    def resolve(
        self,
        params: Dict[str, np.ndarray],
        contacts: List[ContactPoint],
        smplx_wrapper: Optional['SMPLXWrapper'] = None,
        collision_checker: Optional['CollisionChecker'] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Resolve collisions by pushing parts apart.

        Strategy:
        1. For each contact, determine which joints control each body part
        2. Compute a rotation adjustment that moves parts apart
        3. Apply adjustments to body_pose
        4. Optionally iterate with collision_checker feedback

        Args:
            params: SMPL-X parameters
            contacts: Contact points from collision detection
            smplx_wrapper: For forward kinematics (enables iterative)
            collision_checker: For checking if collisions resolved (enables iterative)

        Returns:
            Corrected parameters with (hopefully) no collisions
        """
        if not contacts:
            return params

        # Make a copy to avoid modifying input
        result = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in params.items()}

        # Handle batched body_pose
        body_pose = result.get('body_pose')
        if body_pose is None:
            return result

        was_1d = body_pose.ndim == 1
        if was_1d:
            body_pose = body_pose.reshape(1, -1)

        # Single-shot resolution (no iteration)
        if smplx_wrapper is None or collision_checker is None:
            body_pose = self._apply_resolution_step(body_pose, contacts)
            result['body_pose'] = body_pose.squeeze() if was_1d else body_pose
            return result

        # Iterative resolution with feedback
        for iteration in range(self.max_iterations):
            # Apply resolution step
            body_pose = self._apply_resolution_step(body_pose, contacts)
            result['body_pose'] = body_pose

            # Check if resolved
            output = smplx_wrapper.forward(**result, return_vertices=False)
            collision_checker.update_from_joints(output['joints'])
            new_contacts = collision_checker.get_contact_points()

            if not new_contacts:
                break

            # Check convergence
            max_depth = max(c.depth for c in new_contacts)
            if max_depth < self.convergence_threshold:
                break

            contacts = new_contacts

        result['body_pose'] = body_pose.squeeze() if was_1d else body_pose
        return result

    def _apply_resolution_step(
        self,
        body_pose: np.ndarray,
        contacts: List[ContactPoint],
    ) -> np.ndarray:
        """
        Apply a single resolution step to body_pose.

        Args:
            body_pose: Array of shape (batch, 63) with body joint angles
            contacts: Contact points to resolve

        Returns:
            Adjusted body_pose
        """
        body_pose = body_pose.copy()
        batch_size = body_pose.shape[0]

        for contact in contacts:
            # Get controlling joints for both body parts
            joints_a = get_controlling_joints_for_part(contact.body_part_a)
            joints_b = get_controlling_joints_for_part(contact.body_part_b)

            # Determine which joints to adjust
            # Prefer adjusting distal joints (closer to end effectors)
            # This preserves global pose while resolving local collisions
            all_joints = self._prioritize_joints(joints_a, joints_b)

            # Compute adjustment magnitude based on penetration depth
            magnitude = min(contact.depth * 2, self.step_size)

            for joint_name in all_joints:
                if joint_name not in SMPLX_BODY_JOINTS:
                    continue

                # Get joint index in body_pose
                # body_pose excludes pelvis (index 0), so subtract 1
                joint_idx = SMPLX_BODY_JOINTS[joint_name]
                if joint_idx == 0:  # Skip pelvis
                    continue

                pose_idx = (joint_idx - 1) * 3

                # Compute push direction in joint space
                # This is a simplified approximation - in practice, we'd need
                # the Jacobian to properly map world-space push to joint angles
                delta = self._compute_joint_delta(
                    joint_name,
                    contact,
                    magnitude,
                    joints_a,
                    joints_b,
                )

                # Apply delta to all samples in batch
                for b in range(batch_size):
                    current_angles = body_pose[b, pose_idx:pose_idx + 3]
                    new_angles = current_angles + delta

                    # Clamp to joint limits if requested
                    if self.respect_limits:
                        new_angles = clamp_joint_angles(joint_name, new_angles)

                    body_pose[b, pose_idx:pose_idx + 3] = new_angles

        return body_pose

    def _prioritize_joints(
        self,
        joints_a: List[str],
        joints_b: List[str],
    ) -> List[str]:
        """
        Prioritize joints for adjustment.

        Prefer distal joints (wrist > elbow > shoulder) as they have
        more local effect and preserve global pose.
        """
        # Priority order: more distal first
        priority = {
            'left_wrist': 0, 'right_wrist': 0,
            'left_elbow': 1, 'right_elbow': 1,
            'left_shoulder': 2, 'right_shoulder': 2,
            'left_ankle': 0, 'right_ankle': 0,
            'left_knee': 1, 'right_knee': 1,
            'left_hip': 2, 'right_hip': 2,
            'neck': 3, 'head': 0,
            'spine3': 4, 'spine2': 5, 'spine1': 6,
        }

        all_joints = list(set(joints_a + joints_b))
        all_joints.sort(key=lambda j: priority.get(j, 10))

        # Return top 2-3 joints to avoid over-correction
        return all_joints[:3]

    def _compute_joint_delta(
        self,
        joint_name: str,
        contact: ContactPoint,
        magnitude: float,
        joints_a: List[str],
        joints_b: List[str],
    ) -> np.ndarray:
        """
        Compute rotation delta for a joint to resolve collision.

        This is a simplified heuristic that works well for common cases.
        For more accurate resolution, we'd need to compute the Jacobian.

        Args:
            joint_name: Name of the joint to adjust
            contact: Contact information
            magnitude: How much to adjust (radians)
            joints_a: Joints controlling part A
            joints_b: Joints controlling part B

        Returns:
            Array of shape (3,) with x, y, z rotation deltas
        """
        # Determine if this joint controls part A or part B
        controls_a = joint_name in joints_a
        controls_b = joint_name in joints_b

        # Sign: positive normal points from A to B
        # If controlling A, move in -normal direction (away from B)
        # If controlling B, move in +normal direction (away from A)
        sign = 1.0 if controls_b else -1.0
        if controls_a and controls_b:
            # Joint affects both - use smaller adjustment
            magnitude *= 0.5

        # Convert contact normal to a joint rotation
        # This is a simplification - proper solution would use Jacobian
        normal = contact.normal

        # Heuristic: map world-space normal to joint rotation axes
        # Assumes local joint axes roughly align with world axes
        delta = np.array([
            normal[2] * sign,   # X rotation (around world X) ~ Z normal
            normal[0] * sign,   # Y rotation (around world Y) ~ X normal
            -normal[1] * sign,  # Z rotation (around world Z) ~ -Y normal
        ]) * magnitude

        return delta


class RejectionResolver(CollisionResolver):
    """
    Simple rejection-based resolution.

    If collisions detected, reverts to previous valid pose.
    Fast and conservative - good for real-time applications.
    """

    def __init__(self):
        pass

    def resolve(
        self,
        params: Dict[str, np.ndarray],
        contacts: List[ContactPoint],
        smplx_wrapper: Optional['SMPLXWrapper'] = None,
        collision_checker: Optional['CollisionChecker'] = None,
        prev_params: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        If collisions exist and prev_params available, return prev_params.

        Args:
            params: Current parameters (potentially colliding)
            contacts: Detected contacts
            smplx_wrapper: Not used
            collision_checker: Not used
            prev_params: Previous valid parameters to fall back to

        Returns:
            prev_params if colliding, otherwise params
        """
        if contacts and prev_params is not None:
            return prev_params
        return params


class InterpolationResolver(CollisionResolver):
    """
    Resolution by interpolating toward previous valid pose.

    Binary search to find the largest blend factor that avoids collision.
    """

    def __init__(
        self,
        max_iterations: int = 5,
    ):
        """
        Initialize interpolation resolver.

        Args:
            max_iterations: Maximum binary search iterations
        """
        self.max_iterations = max_iterations

    def resolve(
        self,
        params: Dict[str, np.ndarray],
        contacts: List[ContactPoint],
        smplx_wrapper: Optional['SMPLXWrapper'] = None,
        collision_checker: Optional['CollisionChecker'] = None,
        prev_params: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Find interpolation factor that avoids collision.

        Args:
            params: Current (colliding) parameters
            contacts: Detected contacts
            smplx_wrapper: Required for forward kinematics
            collision_checker: Required for collision checking
            prev_params: Previous valid parameters

        Returns:
            Interpolated parameters that don't collide
        """
        if not contacts:
            return params

        if prev_params is None or smplx_wrapper is None or collision_checker is None:
            # Fall back to gradient resolver
            gradient = GradientResolver()
            return gradient.resolve(params, contacts)

        # Binary search for valid interpolation factor
        low, high = 0.0, 1.0
        best_params = prev_params

        for _ in range(self.max_iterations):
            mid = (low + high) / 2

            # Interpolate parameters
            interp_params = self._interpolate(prev_params, params, mid)

            # Check for collisions
            output = smplx_wrapper.forward(**interp_params, return_vertices=False)
            collision_checker.update_from_joints(output['joints'])

            if collision_checker.has_collisions():
                # Still colliding, reduce factor
                high = mid
            else:
                # No collision, can try higher factor
                low = mid
                best_params = interp_params

        return best_params

    def _interpolate(
        self,
        params_a: Dict[str, np.ndarray],
        params_b: Dict[str, np.ndarray],
        t: float,
    ) -> Dict[str, np.ndarray]:
        """
        Linearly interpolate between two parameter sets.

        Args:
            params_a: Start parameters (t=0)
            params_b: End parameters (t=1)
            t: Interpolation factor [0, 1]

        Returns:
            Interpolated parameters
        """
        result = {}
        for key in params_a:
            if isinstance(params_a[key], np.ndarray) and key in params_b:
                result[key] = params_a[key] * (1 - t) + params_b[key] * t
            else:
                result[key] = params_a[key]
        return result


def create_resolver(strategy: str = 'gradient', **kwargs) -> CollisionResolver:
    """
    Factory function to create a collision resolver.

    Args:
        strategy: One of 'gradient', 'rejection', 'interpolation'
        **kwargs: Strategy-specific arguments

    Returns:
        CollisionResolver instance
    """
    resolvers = {
        'gradient': GradientResolver,
        'rejection': RejectionResolver,
        'interpolation': InterpolationResolver,
    }

    if strategy not in resolvers:
        raise ValueError(f"Unknown resolution strategy: {strategy}. "
                        f"Available: {list(resolvers.keys())}")

    return resolvers[strategy](**kwargs)
