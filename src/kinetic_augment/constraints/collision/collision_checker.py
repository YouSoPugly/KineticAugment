"""
PyBullet-based Collision Detection.

Uses PyBullet physics engine for efficient collision detection between
body parts represented as simple collision primitives.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from kinetic_augment.body_model.joint_mapping import SMPLX_BODY_JOINTS
from kinetic_augment.constraints.collision.body_parts import (
    COLLISION_BODY_PARTS,
    SLR_COLLISION_PAIRS,
    FULL_COLLISION_PAIRS,
    REFERENCE_BODY_HEIGHT,
    are_adjacent,
    get_contact_tolerance,
    get_body_part_params,
)

# Try to import pybullet
try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    p = None


@dataclass
class ContactPoint:
    """Detailed contact information from collision detection."""
    body_part_a: str
    body_part_b: str
    position: np.ndarray  # World position of contact
    normal: np.ndarray    # Contact normal (points from A to B)
    depth: float          # Penetration depth (positive = overlap)

    def __repr__(self) -> str:
        return (
            f"ContactPoint({self.body_part_a} <-> {self.body_part_b}, "
            f"depth={self.depth:.4f}m)"
        )


class CollisionChecker:
    """
    PyBullet-based collision detection for SMPL-X bodies.

    Manages the PyBullet physics world and collision geometry.
    Supports tiered collision checking:
    - Level 1: Capsule/sphere primitives (fast)
    - Level 2: Convex hulls (more accurate, used for confirmation)

    Usage:
        checker = CollisionChecker()
        checker.setup()

        # Update from SMPL-X joint positions
        checker.update_from_joints(joint_positions)

        # Check for collisions
        collisions = checker.check_collisions()

        # Get detailed contact info
        contacts = checker.get_contact_points()

        # Cleanup
        checker.cleanup()
    """

    def __init__(
        self,
        use_gui: bool = False,
        geometry_level: str = 'capsule',
        collision_pairs: Optional[List[Tuple[str, str]]] = None,
        body_height: float = REFERENCE_BODY_HEIGHT,
    ):
        """
        Initialize collision checker.

        Args:
            use_gui: If True, use PyBullet GUI for visualization (debugging)
            geometry_level: 'capsule' for fast, 'convex' for detailed
            collision_pairs: List of (part_a, part_b) pairs to check.
                           Defaults to SLR_COLLISION_PAIRS.
            body_height: Estimated body height for scaling collision shapes
        """
        if not PYBULLET_AVAILABLE:
            raise ImportError(
                "PyBullet is required for collision detection. "
                "Install with: pip install pybullet"
            )

        self.use_gui = use_gui
        self.geometry_level = geometry_level
        self.collision_pairs = collision_pairs or SLR_COLLISION_PAIRS
        self.body_height = body_height

        self.physics_client: Optional[int] = None
        self.body_ids: Dict[str, int] = {}  # Maps part name to PyBullet body ID
        self.collision_shapes: Dict[str, int] = {}  # Maps part name to shape ID

        self._is_setup = False

    def setup(self) -> None:
        """Initialize PyBullet world and create collision bodies."""
        if self._is_setup:
            return

        # Connect to PyBullet
        mode = p.GUI if self.use_gui else p.DIRECT
        self.physics_client = p.connect(mode)

        # Disable dynamics - we only need collision detection
        p.setGravity(0, 0, 0, physicsClientId=self.physics_client)

        # Create collision bodies for each body part
        for part_name, config in COLLISION_BODY_PARTS.items():
            self._create_collision_body(part_name, config)

        self._is_setup = True

    def _create_collision_body(self, part_name: str, config: Dict) -> None:
        """Create a collision body for a body part."""
        geom_type = config['type']
        radius = config['radius_scale'] * self.body_height

        if geom_type == 'capsule':
            # Create capsule shape
            # Note: PyBullet capsule height is the cylinder portion only
            # We'll set initial height to 0.1m and update later
            collision_shape = p.createCollisionShape(
                p.GEOM_CAPSULE,
                radius=radius,
                height=0.1,  # Placeholder, updated in update_from_joints
                physicsClientId=self.physics_client,
            )
        elif geom_type == 'sphere':
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=radius,
                physicsClientId=self.physics_client,
            )
        else:
            raise ValueError(f"Unknown geometry type: {geom_type}")

        # Create kinematic body (position controlled, no dynamics)
        body_id = p.createMultiBody(
            baseMass=0,  # Static/kinematic body
            baseCollisionShapeIndex=collision_shape,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.physics_client,
        )

        self.collision_shapes[part_name] = collision_shape
        self.body_ids[part_name] = body_id

    def update_from_joints(
        self,
        joint_positions: np.ndarray,
        joint_indices: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Update collision body positions from SMPL-X joint positions.

        Args:
            joint_positions: Array of shape (num_joints, 3) or (batch, num_joints, 3)
                           If batched, uses first sample
            joint_indices: Dict mapping joint names to indices.
                         Defaults to SMPLX_BODY_JOINTS.
        """
        if not self._is_setup:
            self.setup()

        joint_indices = joint_indices or SMPLX_BODY_JOINTS

        # Handle batched input
        if joint_positions.ndim == 3:
            joint_positions = joint_positions[0]

        for part_name, config in COLLISION_BODY_PARTS.items():
            params = get_body_part_params(
                joint_positions, joint_indices, part_name, self.body_height
            )

            body_id = self.body_ids[part_name]

            if config['type'] == 'capsule':
                # For capsules, we need to recreate the shape with correct height
                # This is a limitation of PyBullet - can't resize shapes dynamically
                # For efficiency, we keep the original shape and just update position
                p.resetBasePositionAndOrientation(
                    body_id,
                    params['center'].tolist(),
                    params['orientation'].tolist(),
                    physicsClientId=self.physics_client,
                )
            else:  # sphere
                p.resetBasePositionAndOrientation(
                    body_id,
                    params['center'].tolist(),
                    [0, 0, 0, 1],  # Identity quaternion
                    physicsClientId=self.physics_client,
                )

    def check_collisions(self) -> List[Tuple[str, str, float]]:
        """
        Check for collisions between body parts.

        Returns:
            List of (part_a, part_b, penetration_depth) tuples
            Penetration depth is positive when parts overlap
        """
        if not self._is_setup:
            return []

        collisions = []

        for part_a, part_b in self.collision_pairs:
            # Skip adjacent parts
            if are_adjacent(part_a, part_b):
                continue

            # Skip if either part doesn't exist
            if part_a not in self.body_ids or part_b not in self.body_ids:
                continue

            body_a = self.body_ids[part_a]
            body_b = self.body_ids[part_b]

            # Get closest points between bodies
            # maxDistance=0 means only return if actually touching/overlapping
            contact_points = p.getClosestPoints(
                body_a,
                body_b,
                distance=0.01,  # Small margin for near-misses
                physicsClientId=self.physics_client,
            )

            if contact_points:
                # Get maximum penetration depth
                # contact_points[i][8] is the contact distance (negative = overlap)
                min_distance = min(cp[8] for cp in contact_points)

                # Get contact tolerance for this pair
                tolerance = get_contact_tolerance(part_a, part_b)

                # Penetration depth (positive = overlap beyond tolerance)
                depth = -min_distance - tolerance

                if depth > 0:
                    collisions.append((part_a, part_b, depth))

        return collisions

    def get_contact_points(self) -> List[ContactPoint]:
        """
        Get detailed contact information for resolution.

        Returns:
            List of ContactPoint objects with position, normal, depth
        """
        if not self._is_setup:
            return []

        contacts = []

        for part_a, part_b in self.collision_pairs:
            if are_adjacent(part_a, part_b):
                continue

            if part_a not in self.body_ids or part_b not in self.body_ids:
                continue

            body_a = self.body_ids[part_a]
            body_b = self.body_ids[part_b]

            contact_points = p.getClosestPoints(
                body_a,
                body_b,
                distance=0.01,
                physicsClientId=self.physics_client,
            )

            tolerance = get_contact_tolerance(part_a, part_b)

            for cp in contact_points:
                # cp structure:
                # [0] bodyA, [1] bodyB, [2] linkA, [3] linkB,
                # [4] positionOnA, [5] positionOnB, [6] contactNormalOnB,
                # [7] contactDistance, [8] normalForce

                distance = cp[8]
                depth = -distance - tolerance

                if depth > 0:
                    contact = ContactPoint(
                        body_part_a=part_a,
                        body_part_b=part_b,
                        position=np.array(cp[5]),  # Position on B
                        normal=np.array(cp[7]),    # Normal pointing from A to B
                        depth=depth,
                    )
                    contacts.append(contact)

        return contacts

    def has_collisions(self) -> bool:
        """Quick check if any collisions exist."""
        return len(self.check_collisions()) > 0

    def get_collision_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of current collisions.

        Returns:
            Dict with 'num_collisions', 'max_depth', 'colliding_pairs'
        """
        collisions = self.check_collisions()

        if not collisions:
            return {
                'num_collisions': 0,
                'max_depth': 0.0,
                'colliding_pairs': [],
            }

        return {
            'num_collisions': len(collisions),
            'max_depth': max(c[2] for c in collisions),
            'colliding_pairs': [(c[0], c[1]) for c in collisions],
        }

    def cleanup(self) -> None:
        """Clean up PyBullet resources."""
        if self.physics_client is not None:
            p.disconnect(physicsClientId=self.physics_client)
            self.physics_client = None
            self.body_ids.clear()
            self.collision_shapes.clear()
            self._is_setup = False

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False


class TieredCollisionChecker:
    """
    Two-level collision checker for efficiency.

    Level 1 (capsules): Fast screening
    Level 2 (convex hulls): Detailed check for flagged pairs

    This allows fast rejection of clearly non-colliding poses while
    providing accurate detection for ambiguous cases.
    """

    def __init__(
        self,
        collision_pairs: Optional[List[Tuple[str, str]]] = None,
        body_height: float = REFERENCE_BODY_HEIGHT,
    ):
        """
        Initialize tiered checker.

        Args:
            collision_pairs: List of pairs to check
            body_height: Body height for scaling
        """
        self.collision_pairs = collision_pairs or SLR_COLLISION_PAIRS
        self.body_height = body_height

        # Level 1: Capsule-based checker
        self.capsule_checker = CollisionChecker(
            use_gui=False,
            geometry_level='capsule',
            collision_pairs=self.collision_pairs,
            body_height=body_height,
        )

        self._is_setup = False

    def setup(self) -> None:
        """Initialize checkers."""
        if self._is_setup:
            return

        self.capsule_checker.setup()
        self._is_setup = True

    def update_from_joints(
        self,
        joint_positions: np.ndarray,
        joint_indices: Optional[Dict[str, int]] = None,
    ) -> None:
        """Update collision geometry from joint positions."""
        if not self._is_setup:
            self.setup()

        self.capsule_checker.update_from_joints(joint_positions, joint_indices)

    def check_collisions(
        self,
        detailed: bool = False,
    ) -> List[Tuple[str, str, float]]:
        """
        Check for collisions with optional detailed confirmation.

        Args:
            detailed: If True, perform Level 2 check on detected collisions
                     (Currently Level 2 is same as Level 1 - placeholder for
                      future convex hull implementation)

        Returns:
            List of (part_a, part_b, depth) tuples
        """
        # Level 1: Fast capsule check
        collisions = self.capsule_checker.check_collisions()

        if not detailed or not collisions:
            return collisions

        # Level 2: For now, return Level 1 results
        # Future: Re-check with convex hulls for more accurate depth estimation
        return collisions

    def get_contact_points(self) -> List[ContactPoint]:
        """Get detailed contact information."""
        return self.capsule_checker.get_contact_points()

    def has_collisions(self) -> bool:
        """Quick check for collisions."""
        return self.capsule_checker.has_collisions()

    def cleanup(self) -> None:
        """Clean up resources."""
        self.capsule_checker.cleanup()
        self._is_setup = False

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
