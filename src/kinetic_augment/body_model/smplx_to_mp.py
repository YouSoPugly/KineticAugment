"""
SMPL-X to MediaPipe Projection Module.

This module converts SMPL-X model output (joint positions) to MediaPipe
Holistic landmark format.

The projection process:
1. Run SMPL-X forward kinematics to get joint positions
2. Map SMPL-X joints to corresponding MediaPipe landmarks
3. Interpolate/derive landmarks that don't have direct SMPL-X equivalents
4. Optionally normalize to MediaPipe coordinate conventions

Example:
    >>> from kinetic_augment.body_model import get_smplx_wrapper, get_smplx_to_mp
    >>> wrapper = get_smplx_wrapper()(model_path="models/smplx")
    >>> projector = get_smplx_to_mp()(wrapper)
    >>> mp_landmarks = projector.project(smplx_params)
"""

from typing import Dict, Optional, Tuple
import numpy as np

from kinetic_augment.body_model.joint_mapping import (
    SMPLX_BODY_JOINTS,
    SMPLX_EXTRA_JOINTS,
    SMPLX_LEFT_HAND_START,
    SMPLX_RIGHT_HAND_START,
    SMPLX_HAND_JOINTS,
    MEDIAPIPE_POSE_LANDMARKS,
    MEDIAPIPE_HAND_LANDMARKS,
    SMPLX_TO_MEDIAPIPE_POSE,
    SMPLX_TO_MEDIAPIPE_HAND,
)
from kinetic_augment.utils.data_formats import (
    NUM_POSE_LANDMARKS,
    NUM_FACE_LANDMARKS,
    NUM_HAND_LANDMARKS,
    TOTAL_LANDMARKS,
)


class SMPLXToMediaPipe:
    """
    Projects SMPL-X output to MediaPipe landmark format.

    Handles the mapping from SMPL-X joint positions to MediaPipe Holistic
    landmarks, including:
    - Direct joint correspondences (shoulders, elbows, wrists, etc.)
    - Derived landmarks (nose, eyes, ears from head orientation)
    - Hand joint projection
    - Optional face landmark synthesis

    Attributes:
        smplx_wrapper: The SMPL-X model wrapper
        include_face: Whether to generate face landmarks
        include_hands: Whether to generate hand landmarks
    """

    def __init__(
        self,
        smplx_wrapper,
        include_face: bool = True,
        include_hands: bool = True,
        normalize_output: bool = True,
    ):
        """
        Initialize the projector.

        Args:
            smplx_wrapper: Initialized SMPLXWrapper instance
            include_face: Include face landmarks in output
            include_hands: Include hand landmarks in output
            normalize_output: Normalize to MediaPipe coordinate conventions
        """
        self.smplx = smplx_wrapper
        self.include_face = include_face
        self.include_hands = include_hands
        self.normalize = normalize_output

        # Pre-compute head-relative landmark offsets (approximate)
        # These are used to derive face landmarks from head joint position
        self._init_head_offsets()

    def _init_head_offsets(self):
        """Initialize offsets for deriving face landmarks from head joint."""
        # Offsets are in local head coordinates
        # Based on approximate human head proportions
        self.head_offsets = {
            'nose': np.array([0.0, 0.02, 0.12]),
            'left_eye': np.array([-0.03, 0.04, 0.10]),
            'right_eye': np.array([0.03, 0.04, 0.10]),
            'left_eye_inner': np.array([-0.015, 0.04, 0.10]),
            'right_eye_inner': np.array([0.015, 0.04, 0.10]),
            'left_eye_outer': np.array([-0.045, 0.04, 0.09]),
            'right_eye_outer': np.array([0.045, 0.04, 0.09]),
            'left_ear': np.array([-0.08, 0.0, 0.0]),
            'right_ear': np.array([0.08, 0.0, 0.0]),
            'mouth_left': np.array([-0.02, -0.02, 0.10]),
            'mouth_right': np.array([0.02, -0.02, 0.10]),
        }

        # Foot offsets from ankle (approximate)
        self.foot_offsets = {
            'heel': np.array([-0.05, -0.08, 0.0]),
            'foot_index': np.array([0.12, -0.08, 0.0]),
        }

        # Hand landmark offsets from wrist (simplified)
        self.hand_offsets = {
            'thumb': np.array([0.04, 0.03, 0.02]),
            'index': np.array([0.0, 0.08, 0.01]),
            'pinky': np.array([-0.03, 0.07, -0.01]),
        }

    def project(
        self,
        smplx_params: Dict[str, np.ndarray],
        return_full_frame: bool = True,
    ) -> np.ndarray:
        """
        Project SMPL-X parameters to MediaPipe landmarks.

        Args:
            smplx_params: Dictionary of SMPL-X parameters
            return_full_frame: If True, return full 543-landmark array
                              If False, return only pose landmarks (33)

        Returns:
            MediaPipe landmarks, shape (batch, 543, 3) or (batch, 33, 3)
        """
        # Run SMPL-X forward kinematics
        output = self.smplx.forward(**smplx_params, return_vertices=False)
        joints = output['joints']  # (batch, num_joints, 3)

        batch_size = joints.shape[0]

        # Initialize output array
        if return_full_frame:
            landmarks = np.zeros((batch_size, TOTAL_LANDMARKS, 3), dtype=np.float32)
        else:
            landmarks = np.zeros((batch_size, NUM_POSE_LANDMARKS, 3), dtype=np.float32)

        # Map pose landmarks
        landmarks[:, :NUM_POSE_LANDMARKS, :] = self._map_pose_landmarks(joints)

        # Add face and hand landmarks if requested
        if return_full_frame:
            if self.include_face:
                face_start = NUM_POSE_LANDMARKS
                face_end = face_start + NUM_FACE_LANDMARKS
                landmarks[:, face_start:face_end, :] = self._generate_face_landmarks(
                    joints, batch_size
                )

            if self.include_hands:
                lh_start = NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS
                lh_end = lh_start + NUM_HAND_LANDMARKS
                rh_start = lh_end
                rh_end = rh_start + NUM_HAND_LANDMARKS

                landmarks[:, lh_start:lh_end, :] = self._map_hand_landmarks(
                    joints, 'left'
                )
                landmarks[:, rh_start:rh_end, :] = self._map_hand_landmarks(
                    joints, 'right'
                )

        # Normalize if requested
        if self.normalize:
            landmarks = self._normalize_landmarks(landmarks)

        return landmarks

    def _map_pose_landmarks(self, joints: np.ndarray) -> np.ndarray:
        """
        Map SMPL-X joints to MediaPipe pose landmarks.

        Args:
            joints: SMPL-X joint positions (batch, num_joints, 3)

        Returns:
            Pose landmarks (batch, 33, 3)
        """
        batch_size = joints.shape[0]
        pose = np.zeros((batch_size, NUM_POSE_LANDMARKS, 3), dtype=np.float32)

        # Direct mappings
        for smplx_name, mp_name in SMPLX_TO_MEDIAPIPE_POSE.items():
            if smplx_name in SMPLX_BODY_JOINTS:
                smplx_idx = SMPLX_BODY_JOINTS[smplx_name]
            elif smplx_name in SMPLX_EXTRA_JOINTS:
                smplx_idx = SMPLX_EXTRA_JOINTS[smplx_name]
            else:
                continue

            if mp_name in MEDIAPIPE_POSE_LANDMARKS:
                mp_idx = MEDIAPIPE_POSE_LANDMARKS[mp_name]
                pose[:, mp_idx, :] = joints[:, smplx_idx, :]

        # Derive head-based landmarks (nose, eyes, ears, mouth)
        head_pos = joints[:, SMPLX_BODY_JOINTS['head'], :]
        neck_pos = joints[:, SMPLX_BODY_JOINTS['neck'], :]

        # Compute head orientation (simplified)
        head_forward = self._compute_head_forward(joints)
        head_right = self._compute_head_right(joints)
        head_up = np.cross(head_forward, head_right)

        for name, offset in self.head_offsets.items():
            if name in MEDIAPIPE_POSE_LANDMARKS:
                mp_idx = MEDIAPIPE_POSE_LANDMARKS[name]
                # Transform offset from local to world coordinates
                world_offset = (
                    offset[0] * head_right +
                    offset[1] * head_up +
                    offset[2] * head_forward
                )
                pose[:, mp_idx, :] = head_pos + world_offset

        # Derive foot landmarks from ankles
        for side in ['left', 'right']:
            ankle_pos = joints[:, SMPLX_BODY_JOINTS[f'{side}_ankle'], :]
            foot_pos = joints[:, SMPLX_BODY_JOINTS[f'{side}_foot'], :]

            # Foot direction
            foot_dir = foot_pos - ankle_pos
            foot_dir = foot_dir / (np.linalg.norm(foot_dir, axis=1, keepdims=True) + 1e-8)

            # Heel
            heel_idx = MEDIAPIPE_POSE_LANDMARKS[f'{side}_heel']
            pose[:, heel_idx, :] = ankle_pos + self.foot_offsets['heel']

            # Foot index (toe)
            toe_idx = MEDIAPIPE_POSE_LANDMARKS[f'{side}_foot_index']
            pose[:, toe_idx, :] = foot_pos

        # Derive basic hand landmarks from wrist
        for side in ['left', 'right']:
            wrist_pos = joints[:, SMPLX_BODY_JOINTS[f'{side}_wrist'], :]
            elbow_pos = joints[:, SMPLX_BODY_JOINTS[f'{side}_elbow'], :]

            # Hand direction (from wrist, opposite to forearm)
            forearm = wrist_pos - elbow_pos
            forearm_norm = forearm / (np.linalg.norm(forearm, axis=1, keepdims=True) + 1e-8)

            # Simple offsets along forearm direction
            thumb_idx = MEDIAPIPE_POSE_LANDMARKS[f'{side}_thumb']
            index_idx = MEDIAPIPE_POSE_LANDMARKS[f'{side}_index']
            pinky_idx = MEDIAPIPE_POSE_LANDMARKS[f'{side}_pinky']

            pose[:, thumb_idx, :] = wrist_pos + forearm_norm * 0.05
            pose[:, index_idx, :] = wrist_pos + forearm_norm * 0.08
            pose[:, pinky_idx, :] = wrist_pos + forearm_norm * 0.07

        return pose

    def _compute_head_forward(self, joints: np.ndarray) -> np.ndarray:
        """Compute head forward direction."""
        head = joints[:, SMPLX_BODY_JOINTS['head'], :]
        neck = joints[:, SMPLX_BODY_JOINTS['neck'], :]

        # Approximate forward as perpendicular to neck-head and left-right shoulder
        up = head - neck
        up = up / (np.linalg.norm(up, axis=1, keepdims=True) + 1e-8)

        # Use shoulder line for right direction
        right = self._compute_head_right(joints)

        # Forward = right × up
        forward = np.cross(right, up)
        forward = forward / (np.linalg.norm(forward, axis=1, keepdims=True) + 1e-8)

        return forward

    def _compute_head_right(self, joints: np.ndarray) -> np.ndarray:
        """Compute head right direction (from left to right shoulder)."""
        left_shoulder = joints[:, SMPLX_BODY_JOINTS['left_shoulder'], :]
        right_shoulder = joints[:, SMPLX_BODY_JOINTS['right_shoulder'], :]

        right = right_shoulder - left_shoulder
        right = right / (np.linalg.norm(right, axis=1, keepdims=True) + 1e-8)

        return right

    def _map_hand_landmarks(
        self, joints: np.ndarray, side: str
    ) -> np.ndarray:
        """
        Map SMPL-X hand joints to MediaPipe hand landmarks.

        Args:
            joints: SMPL-X joint positions (batch, num_joints, 3)
            side: 'left' or 'right'

        Returns:
            Hand landmarks (batch, 21, 3)
        """
        batch_size = joints.shape[0]
        hand = np.zeros((batch_size, NUM_HAND_LANDMARKS, 3), dtype=np.float32)

        # Get hand joint start index
        if side == 'left':
            hand_start = SMPLX_LEFT_HAND_START
        else:
            hand_start = SMPLX_RIGHT_HAND_START

        # Wrist (index 0 in MediaPipe hand)
        wrist_smplx = SMPLX_BODY_JOINTS[f'{side}_wrist']
        hand[:, 0, :] = joints[:, wrist_smplx, :]

        # Map finger joints
        for smplx_name, mp_name in SMPLX_TO_MEDIAPIPE_HAND.items():
            if smplx_name in SMPLX_HAND_JOINTS and mp_name in MEDIAPIPE_HAND_LANDMARKS:
                smplx_idx = hand_start + SMPLX_HAND_JOINTS[smplx_name]
                mp_idx = MEDIAPIPE_HAND_LANDMARKS[mp_name]
                hand[:, mp_idx, :] = joints[:, smplx_idx, :]

        # Derive fingertips (not directly in SMPL-X)
        # Extrapolate from last joint
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            if finger == 'thumb':
                last_joint = 'thumb3'
                prev_joint = 'thumb2'
            else:
                last_joint = f'{finger}3'
                prev_joint = f'{finger}2'

            if last_joint in SMPLX_HAND_JOINTS:
                last_idx = hand_start + SMPLX_HAND_JOINTS[last_joint]
                prev_idx = hand_start + SMPLX_HAND_JOINTS[prev_joint]

                last_pos = joints[:, last_idx, :]
                prev_pos = joints[:, prev_idx, :]

                # Extrapolate tip
                direction = last_pos - prev_pos
                tip_pos = last_pos + direction * 0.5

                tip_name = f'{finger}_tip'
                if tip_name in MEDIAPIPE_HAND_LANDMARKS:
                    tip_idx = MEDIAPIPE_HAND_LANDMARKS[tip_name]
                    hand[:, tip_idx, :] = tip_pos

        return hand

    def _generate_face_landmarks(
        self, joints: np.ndarray, batch_size: int
    ) -> np.ndarray:
        """
        Generate placeholder face landmarks.

        Full face landmark generation would require the FLAME model
        or similar. This generates a simplified placeholder.

        Args:
            joints: SMPL-X joints
            batch_size: Number of samples

        Returns:
            Face landmarks (batch, 468, 3) - mostly zeros with key points
        """
        face = np.zeros((batch_size, NUM_FACE_LANDMARKS, 3), dtype=np.float32)

        # For now, just return zeros
        # A full implementation would use SMPL-X expression parameters
        # and the FLAME face model to generate proper face landmarks

        return face

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to MediaPipe coordinate conventions.

        MediaPipe uses:
        - x, y: normalized to [0, 1] by image dimensions
        - z: depth relative to hip center (smaller = closer)

        Args:
            landmarks: Raw landmarks (batch, num_landmarks, 3)

        Returns:
            Normalized landmarks
        """
        batch_size = landmarks.shape[0]
        normalized = landmarks.copy()

        for b in range(batch_size):
            # Get hip center for reference
            left_hip = landmarks[b, MEDIAPIPE_POSE_LANDMARKS['left_hip'], :]
            right_hip = landmarks[b, MEDIAPIPE_POSE_LANDMARKS['right_hip'], :]
            hip_center = (left_hip + right_hip) / 2

            # Compute bounding box
            valid_mask = np.any(landmarks[b] != 0, axis=1)
            if not valid_mask.any():
                continue

            valid_points = landmarks[b, valid_mask, :]
            min_coords = valid_points.min(axis=0)
            max_coords = valid_points.max(axis=0)
            ranges = max_coords - min_coords
            ranges = np.maximum(ranges, 1e-6)

            # Normalize x, y to [0, 1]
            # Note: MediaPipe has Y increasing downward
            normalized[b, :, 0] = (landmarks[b, :, 0] - min_coords[0]) / ranges[0]
            normalized[b, :, 1] = (landmarks[b, :, 1] - min_coords[1]) / ranges[1]

            # Z: relative to hip center
            normalized[b, :, 2] = landmarks[b, :, 2] - hip_center[2]

        return normalized

    def project_sequence(
        self,
        params_sequence: list,
    ) -> np.ndarray:
        """
        Project a sequence of SMPL-X parameters.

        Args:
            params_sequence: List of SMPL-X parameter dictionaries

        Returns:
            Landmarks sequence (num_frames, num_landmarks, 3)
        """
        results = []

        for params in params_sequence:
            # Ensure batch dimension
            batch_params = {}
            for k, v in params.items():
                if k.startswith('_'):
                    continue
                if v.ndim == 1:
                    batch_params[k] = v[np.newaxis, ...]
                else:
                    batch_params[k] = v

            landmarks = self.project(batch_params)
            results.append(landmarks[0])  # Remove batch dim

        return np.stack(results, axis=0)


def quick_project(
    smplx_wrapper,
    body_pose: np.ndarray,
    left_hand_pose: Optional[np.ndarray] = None,
    right_hand_pose: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Quick utility function to project SMPL-X pose to MediaPipe landmarks.

    Args:
        smplx_wrapper: Initialized SMPLXWrapper
        body_pose: Body pose (63,) or (batch, 63)
        left_hand_pose: Left hand pose (optional)
        right_hand_pose: Right hand pose (optional)

    Returns:
        MediaPipe landmarks
    """
    projector = SMPLXToMediaPipe(smplx_wrapper)

    # Build parameter dict
    if body_pose.ndim == 1:
        body_pose = body_pose[np.newaxis, ...]

    params = smplx_wrapper.params_to_dict(
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
    )

    return projector.project(params)
