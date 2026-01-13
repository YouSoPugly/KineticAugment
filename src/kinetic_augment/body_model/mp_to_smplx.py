"""
MediaPipe to SMPL-X Fitting Module.

This module provides optimization-based fitting of SMPL-X parameters
to MediaPipe Holistic landmarks.

The fitting process:
1. Initialize SMPL-X parameters (default pose or from previous frame)
2. Compute loss between SMPL-X joints and MediaPipe landmarks
3. Optimize parameters using gradient descent
4. Return fitted SMPL-X parameters

Example:
    >>> from kinetic_augment.body_model import get_smplx_wrapper, get_mp_to_smplx
    >>> wrapper = get_smplx_wrapper()(model_path="models/smplx")
    >>> fitter = get_mp_to_smplx()(wrapper)
    >>> smplx_params = fitter.fit(mediapipe_landmarks)
"""

from __future__ import annotations  # Enable lazy evaluation of type hints

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm

# Check for torch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from kinetic_augment.body_model.joint_mapping import (
    MEDIAPIPE_POSE_LANDMARKS,
    SMPLX_BODY_JOINTS,
    SMPLX_TO_MEDIAPIPE_POSE,
    get_corresponding_joints,
)
from kinetic_augment.utils.data_formats import (
    NUM_POSE_LANDMARKS,
    NUM_HAND_LANDMARKS,
    LH_START,
    RH_START,
)


class MediaPipeToSMPLX:
    """
    Fits SMPL-X parameters to MediaPipe landmarks.

    Uses optimization to find SMPL-X body pose, hand pose, and optionally
    shape parameters that minimize the distance between SMPL-X joint
    positions and MediaPipe landmarks.

    Attributes:
        smplx_wrapper: The SMPL-X model wrapper
        joint_mapping: Mapping from SMPL-X joints to MediaPipe landmarks
        device: Torch device
    """

    def __init__(
        self,
        smplx_wrapper,
        learning_rate: float = 0.01,
        num_iterations: int = 100,
        fit_shape: bool = False,
        fit_hands: bool = True,
        regularization_weight: float = 0.01,
        verbose: bool = False,
    ):
        """
        Initialize the fitter.

        Args:
            smplx_wrapper: Initialized SMPLXWrapper instance
            learning_rate: Optimization learning rate
            num_iterations: Number of optimization iterations
            fit_shape: Whether to optimize body shape (betas)
            fit_hands: Whether to optimize hand pose
            regularization_weight: Weight for pose regularization
            verbose: Print progress during fitting
        """
        if not TORCH_AVAILABLE:
            raise ImportError("MediaPipeToSMPLX requires PyTorch. Install with: pip install torch")

        self.smplx = smplx_wrapper
        self.device = smplx_wrapper.device
        self.lr = learning_rate
        self.num_iters = num_iterations
        self.fit_shape = fit_shape
        self.fit_hands = fit_hands
        self.reg_weight = regularization_weight
        self.verbose = verbose

        # Build joint correspondence mapping
        self._build_joint_mapping()

        # Reference bone lengths (computed from first fit, used for scaling)
        self._reference_scale = None

    def _build_joint_mapping(self):
        """Build the mapping between SMPL-X and MediaPipe joints."""
        self.smplx_to_mp = get_corresponding_joints('smplx')

        # Create index arrays for efficient loss computation
        smplx_indices = []
        mp_indices = []

        for smplx_idx, mp_idx in self.smplx_to_mp.items():
            smplx_indices.append(smplx_idx)
            mp_indices.append(mp_idx)

        self.smplx_joint_indices = smplx_indices
        self.mp_landmark_indices = mp_indices

        # Weights for different joints (higher = more important)
        self.joint_weights = self._compute_joint_weights()

    def _compute_joint_weights(self) -> np.ndarray:
        """
        Compute importance weights for each landmark correspondence.

        Wrists and hands get higher weight for sign language applications.
        """
        weights = np.ones(len(self.mp_landmark_indices))

        # Higher weight for end effectors (hands, feet)
        high_weight_landmarks = {
            MEDIAPIPE_POSE_LANDMARKS['left_wrist'],
            MEDIAPIPE_POSE_LANDMARKS['right_wrist'],
            MEDIAPIPE_POSE_LANDMARKS['left_ankle'],
            MEDIAPIPE_POSE_LANDMARKS['right_ankle'],
        }

        for i, mp_idx in enumerate(self.mp_landmark_indices):
            if mp_idx in high_weight_landmarks:
                weights[i] = 2.0

        return weights

    def fit(
        self,
        landmarks: np.ndarray,
        initial_params: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Fit SMPL-X parameters to MediaPipe landmarks.

        Args:
            landmarks: MediaPipe landmarks, shape (543, 3) or (num_frames, 543, 3)
            initial_params: Optional initial SMPL-X parameters

        Returns:
            Dictionary of fitted SMPL-X parameters
        """
        # Handle batch dimension
        if landmarks.ndim == 2:
            landmarks = landmarks[np.newaxis, ...]  # Add batch dim
            single_frame = True
        else:
            single_frame = False

        batch_size = landmarks.shape[0]

        # Extract pose landmarks (first 33)
        pose_landmarks = landmarks[:, :NUM_POSE_LANDMARKS, :]

        # Extract hand landmarks if fitting hands
        if self.fit_hands:
            left_hand = landmarks[:, LH_START:LH_START + NUM_HAND_LANDMARKS, :]
            right_hand = landmarks[:, RH_START:RH_START + NUM_HAND_LANDMARKS, :]
        else:
            left_hand = None
            right_hand = None

        # Normalize landmarks to SMPL-X scale
        pose_landmarks_norm, scale_factor = self._normalize_landmarks(pose_landmarks)

        # Initialize parameters
        if initial_params is not None:
            params = {k: torch.from_numpy(v).to(self.device).float()
                     for k, v in initial_params.items()}
        else:
            params = self._initialize_params(batch_size)

        # Set up optimization
        optimizable_params = self._get_optimizable_params(params)
        optimizer = optim.Adam(optimizable_params, lr=self.lr)

        # Convert target landmarks to tensor
        target = torch.from_numpy(pose_landmarks_norm).to(self.device).float()
        weights = torch.from_numpy(self.joint_weights).to(self.device).float()

        # Optimization loop
        iterator = range(self.num_iters)
        if self.verbose:
            iterator = tqdm(iterator, desc="Fitting SMPL-X")

        for i in iterator:
            optimizer.zero_grad()

            # Forward pass through SMPL-X
            output = self.smplx.model(
                betas=params.get('betas'),
                body_pose=params['body_pose'],
                global_orient=params['global_orient'],
                left_hand_pose=params.get('left_hand_pose'),
                right_hand_pose=params.get('right_hand_pose'),
                return_verts=False,
            )

            # Get predicted joint positions
            pred_joints = output.joints  # (batch, num_joints, 3)

            # Compute loss
            loss = self._compute_loss(pred_joints, target, weights, params)

            # Backward pass
            loss.backward()
            optimizer.step()

            if self.verbose and i % 20 == 0:
                tqdm.write(f"  Iter {i}: loss = {loss.item():.6f}")

        # Extract final parameters
        result = self._extract_params(params, scale_factor)

        if single_frame:
            # Remove batch dimension
            result = {k: v[0] for k, v in result.items()}

        return result

    def _normalize_landmarks(
        self, landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize landmarks to SMPL-X scale.

        Centers on pelvis (hip midpoint) and scales based on torso length.

        Args:
            landmarks: Pose landmarks (batch, 33, 3)

        Returns:
            Tuple of (normalized_landmarks, scale_factors)
        """
        batch_size = landmarks.shape[0]

        # Compute hip center (pelvis proxy)
        left_hip_idx = MEDIAPIPE_POSE_LANDMARKS['left_hip']
        right_hip_idx = MEDIAPIPE_POSE_LANDMARKS['right_hip']
        hip_center = (landmarks[:, left_hip_idx] + landmarks[:, right_hip_idx]) / 2

        # Compute shoulder center
        left_shoulder_idx = MEDIAPIPE_POSE_LANDMARKS['left_shoulder']
        right_shoulder_idx = MEDIAPIPE_POSE_LANDMARKS['right_shoulder']
        shoulder_center = (landmarks[:, left_shoulder_idx] + landmarks[:, right_shoulder_idx]) / 2

        # Compute torso length for scaling
        torso_length = np.linalg.norm(shoulder_center - hip_center, axis=1)
        torso_length = np.maximum(torso_length, 1e-6)  # Avoid division by zero

        # SMPL-X torso is approximately 0.5 units
        scale_factors = 0.5 / torso_length

        # Center and scale
        normalized = landmarks.copy()
        for b in range(batch_size):
            normalized[b] = (landmarks[b] - hip_center[b]) * scale_factors[b]

        return normalized, scale_factors

    def _initialize_params(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Initialize SMPL-X parameters."""
        params = {}

        # Body pose (21 joints × 3 = 63)
        params['body_pose'] = torch.zeros(
            batch_size, 63, device=self.device, requires_grad=True
        )

        # Global orientation
        params['global_orient'] = torch.zeros(
            batch_size, 3, device=self.device, requires_grad=True
        )

        # Shape (optional)
        if self.fit_shape:
            params['betas'] = torch.zeros(
                batch_size, self.smplx.num_betas, device=self.device, requires_grad=True
            )
        else:
            params['betas'] = torch.zeros(
                batch_size, self.smplx.num_betas, device=self.device
            )

        # Hands (optional)
        if self.fit_hands:
            params['left_hand_pose'] = torch.zeros(
                batch_size, self.smplx.hand_pose_dim, device=self.device, requires_grad=True
            )
            params['right_hand_pose'] = torch.zeros(
                batch_size, self.smplx.hand_pose_dim, device=self.device, requires_grad=True
            )

        return params

    def _get_optimizable_params(
        self, params: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """Get list of parameters to optimize."""
        opt_params = [params['body_pose'], params['global_orient']]

        if self.fit_shape and params['betas'].requires_grad:
            opt_params.append(params['betas'])

        if self.fit_hands:
            if 'left_hand_pose' in params:
                opt_params.append(params['left_hand_pose'])
            if 'right_hand_pose' in params:
                opt_params.append(params['right_hand_pose'])

        return opt_params

    def _compute_loss(
        self,
        pred_joints: torch.Tensor,
        target_landmarks: torch.Tensor,
        weights: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute fitting loss.

        Loss = landmark_loss + regularization_loss
        """
        # Extract corresponding joints
        pred_subset = pred_joints[:, self.smplx_joint_indices, :]
        target_subset = target_landmarks[:, self.mp_landmark_indices, :]

        # Weighted L2 loss
        diff = pred_subset - target_subset
        landmark_loss = (weights[None, :, None] * diff ** 2).mean()

        # Regularization: prefer poses close to neutral
        reg_loss = self.reg_weight * (
            params['body_pose'] ** 2
        ).mean()

        if self.fit_hands and 'left_hand_pose' in params:
            reg_loss += self.reg_weight * 0.5 * (
                params['left_hand_pose'] ** 2 +
                params['right_hand_pose'] ** 2
            ).mean()

        return landmark_loss + reg_loss

    def _extract_params(
        self,
        params: Dict[str, torch.Tensor],
        scale_factors: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Convert torch parameters to numpy."""
        result = {}

        for key, tensor in params.items():
            if isinstance(tensor, torch.Tensor):
                result[key] = tensor.detach().cpu().numpy()
            else:
                result[key] = tensor

        # Store scale factor for later use
        result['_scale_factor'] = scale_factors

        return result

    def fit_sequence(
        self,
        landmarks_sequence: np.ndarray,
        use_temporal_smoothing: bool = True,
        smoothing_weight: float = 0.1,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Fit SMPL-X to a sequence of frames with temporal consistency.

        Args:
            landmarks_sequence: Landmarks (num_frames, 543, 3)
            use_temporal_smoothing: Use previous frame as initialization
            smoothing_weight: Weight for temporal smoothness term

        Returns:
            List of SMPL-X parameter dictionaries
        """
        num_frames = landmarks_sequence.shape[0]
        results = []

        prev_params = None

        for i in tqdm(range(num_frames), desc="Fitting sequence"):
            frame_landmarks = landmarks_sequence[i]

            # Use previous frame as initialization
            if use_temporal_smoothing and prev_params is not None:
                initial = prev_params
            else:
                initial = None

            # Fit this frame
            params = self.fit(frame_landmarks, initial_params=initial)
            results.append(params)

            prev_params = {
                k: v[np.newaxis, ...] if v.ndim == 1 else v
                for k, v in params.items()
                if not k.startswith('_')
            }

        return results


class SimpleFitter:
    """
    Simplified fitter using analytical methods instead of optimization.

    This is faster but less accurate than the optimization-based approach.
    Useful for quick prototyping or when SMPL-X is not available.
    """

    def __init__(self):
        """Initialize the simple fitter."""
        self._build_joint_mapping()

    def _build_joint_mapping(self):
        """Build mapping from MediaPipe to SMPL-X."""
        self.mp_to_smplx = get_corresponding_joints('mediapipe')

    def fit(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Estimate SMPL-X parameters from landmarks using analytical methods.

        This computes joint angles by analyzing vectors between landmarks.

        Args:
            landmarks: MediaPipe landmarks (543, 3) or (num_frames, 543, 3)

        Returns:
            Estimated SMPL-X parameters (approximate)
        """
        if landmarks.ndim == 2:
            landmarks = landmarks[np.newaxis, ...]

        batch_size = landmarks.shape[0]
        pose_landmarks = landmarks[:, :NUM_POSE_LANDMARKS, :]

        # Initialize output
        body_pose = np.zeros((batch_size, 63), dtype=np.float32)
        global_orient = np.zeros((batch_size, 3), dtype=np.float32)

        for b in range(batch_size):
            # Compute global orientation from hip-shoulder alignment
            global_orient[b] = self._compute_global_orient(pose_landmarks[b])

            # Compute individual joint angles
            body_pose[b] = self._compute_body_pose(pose_landmarks[b])

        return {
            'body_pose': body_pose,
            'global_orient': global_orient,
            'betas': np.zeros((batch_size, 10), dtype=np.float32),
            'left_hand_pose': np.zeros((batch_size, 45), dtype=np.float32),
            'right_hand_pose': np.zeros((batch_size, 45), dtype=np.float32),
        }

    def _compute_global_orient(self, landmarks: np.ndarray) -> np.ndarray:
        """Compute global body orientation from landmarks."""
        # Use hip and shoulder vectors to determine orientation
        left_hip = landmarks[MEDIAPIPE_POSE_LANDMARKS['left_hip']]
        right_hip = landmarks[MEDIAPIPE_POSE_LANDMARKS['right_hip']]
        left_shoulder = landmarks[MEDIAPIPE_POSE_LANDMARKS['left_shoulder']]
        right_shoulder = landmarks[MEDIAPIPE_POSE_LANDMARKS['right_shoulder']]

        # Hip vector (pointing right)
        hip_vec = right_hip - left_hip
        hip_vec = hip_vec / (np.linalg.norm(hip_vec) + 1e-8)

        # Up vector (from hip center to shoulder center)
        hip_center = (left_hip + right_hip) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2
        up_vec = shoulder_center - hip_center
        up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-8)

        # Forward vector (cross product)
        forward_vec = np.cross(hip_vec, up_vec)
        forward_vec = forward_vec / (np.linalg.norm(forward_vec) + 1e-8)

        # Convert to axis-angle (simplified)
        # This is approximate - a full implementation would use proper rotation matrices
        yaw = np.arctan2(forward_vec[0], forward_vec[2])
        pitch = np.arcsin(np.clip(-forward_vec[1], -1, 1))
        roll = np.arctan2(hip_vec[1], np.sqrt(hip_vec[0]**2 + hip_vec[2]**2))

        return np.array([pitch, yaw, roll], dtype=np.float32)

    def _compute_body_pose(self, landmarks: np.ndarray) -> np.ndarray:
        """Compute body joint angles from landmarks."""
        pose = np.zeros(63, dtype=np.float32)

        # This is a simplified implementation
        # A full implementation would compute proper joint angles

        # Example: compute elbow angles
        for side, sign in [('left', -1), ('right', 1)]:
            shoulder = landmarks[MEDIAPIPE_POSE_LANDMARKS[f'{side}_shoulder']]
            elbow = landmarks[MEDIAPIPE_POSE_LANDMARKS[f'{side}_elbow']]
            wrist = landmarks[MEDIAPIPE_POSE_LANDMARKS[f'{side}_wrist']]

            # Upper arm vector
            upper_arm = elbow - shoulder
            upper_arm = upper_arm / (np.linalg.norm(upper_arm) + 1e-8)

            # Forearm vector
            forearm = wrist - elbow
            forearm = forearm / (np.linalg.norm(forearm) + 1e-8)

            # Elbow angle
            elbow_angle = np.arccos(np.clip(np.dot(-upper_arm, forearm), -1, 1))

            # Store in appropriate position
            # (This mapping is simplified - actual SMPL-X joint order may differ)
            elbow_idx = SMPLX_BODY_JOINTS.get(f'{side}_elbow', 0) - 1  # -1 for pelvis offset
            if 0 <= elbow_idx < 21:
                pose[elbow_idx * 3] = elbow_angle

        return pose
