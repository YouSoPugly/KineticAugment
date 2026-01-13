"""
WLASL Dataset Loader for KineticAugment Evaluation.

WLASL (Word-Level American Sign Language) is a large-scale video dataset
for American Sign Language recognition.

Reference: Li et al., "Word-level Deep Sign Language Recognition from Video"
https://github.com/dxli94/WLASL
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from kinetic_augment.evaluation.datasets.base import EvaluationDataset


class WLASLDataset(EvaluationDataset):
    """
    WLASL (Word-Level American Sign Language) dataset loader.

    Supports loading pre-extracted MediaPipe landmarks from WLASL videos.
    The dataset should be preprocessed to extract landmarks before use.

    Directory structure expected:
        wlasl/
        ├── WLASL_v0.3.json          # Annotations
        ├── landmarks/                # Pre-extracted landmarks
        │   ├── 00001.npy
        │   ├── 00002.npy
        │   └── ...
        └── videos/                   # Original videos (optional)
            ├── 00001.mp4
            └── ...

    Example:
        >>> dataset = WLASLDataset(
        ...     root_path='data/wlasl',
        ...     subset='wlasl100',
        ...     split='train',
        ... )
        >>> landmarks, label = dataset[0]
    """

    name = "WLASL"

    SUBSETS = ['wlasl100', 'wlasl300', 'wlasl1000', 'wlasl2000']
    SPLITS = ['train', 'val', 'test']

    def __init__(
        self,
        root_path: Union[str, Path],
        subset: str = 'wlasl100',
        split: str = 'train',
        landmark_dir: str = 'landmarks',
        annotation_file: str = 'WLASL_v0.3.json',
        max_frames: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize WLASL dataset.

        Args:
            root_path: Path to WLASL dataset root
            subset: Dataset subset ('wlasl100', 'wlasl300', 'wlasl1000', 'wlasl2000')
            split: Data split ('train', 'val', 'test')
            landmark_dir: Directory containing pre-extracted landmarks
            annotation_file: Name of annotation JSON file
            max_frames: Maximum frames per sample (truncate if longer)
            transform: Optional transform function
        """
        self.root_path = Path(root_path)
        self.subset = subset
        self.split = split
        self.landmark_dir = self.root_path / landmark_dir
        self.max_frames = max_frames
        self.transform = transform

        if subset not in self.SUBSETS:
            raise ValueError(f"Unknown subset: {subset}. Choose from {self.SUBSETS}")
        if split not in self.SPLITS:
            raise ValueError(f"Unknown split: {split}. Choose from {self.SPLITS}")

        # Load annotations
        self._load_annotations(annotation_file)

    def _load_annotations(self, annotation_file: str) -> None:
        """Load WLASL annotations and filter by subset/split."""
        annotation_path = self.root_path / annotation_file

        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_path}\n"
                f"Please download WLASL annotations from: "
                f"https://github.com/dxli94/WLASL"
            )

        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        # Determine number of classes for subset
        subset_num = int(self.subset.replace('wlasl', ''))

        # Build gloss list and samples
        self.gloss_list: List[str] = []
        self.samples: List[Dict] = []

        for entry in annotations[:subset_num]:
            gloss = entry['gloss']
            gloss_idx = len(self.gloss_list)
            self.gloss_list.append(gloss)

            for instance in entry.get('instances', []):
                # Check split
                instance_split = instance.get('split', 'train')
                if instance_split != self.split:
                    continue

                video_id = instance.get('video_id', '')
                if not video_id:
                    continue

                # Check if landmark file exists
                landmark_path = self.landmark_dir / f"{video_id}.npy"
                if not landmark_path.exists():
                    continue

                self.samples.append({
                    'video_id': video_id,
                    'gloss': gloss,
                    'label': gloss_idx,
                    'landmark_path': landmark_path,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (landmarks, label)
        """
        sample = self.samples[idx]

        # Load landmarks
        landmarks = np.load(sample['landmark_path'])

        # Ensure correct shape
        if landmarks.ndim == 2:
            # Assume (frames, features) -> reshape to (frames, landmarks, 3)
            num_frames = landmarks.shape[0]
            landmarks = landmarks.reshape(num_frames, -1, 3)

        # Truncate if needed
        if self.max_frames is not None and landmarks.shape[0] > self.max_frames:
            landmarks = landmarks[:self.max_frames]

        # Apply transform if provided
        if self.transform is not None:
            landmarks = self.transform(landmarks)

        return landmarks.astype(np.float32), sample['label']

    def get_class_names(self) -> List[str]:
        return self.gloss_list.copy()

    def get_sample_info(self, idx: int) -> dict:
        """Get extended sample information."""
        sample = self.samples[idx]
        landmarks, label = self[idx]

        return {
            'index': idx,
            'video_id': sample['video_id'],
            'gloss': sample['gloss'],
            'label': label,
            'num_frames': landmarks.shape[0],
            'num_landmarks': landmarks.shape[1],
        }


class WLASLPreprocessor:
    """
    Preprocess WLASL videos to extract MediaPipe landmarks.

    Batch processes videos and saves landmarks as .npy files.
    """

    def __init__(
        self,
        video_dir: Union[str, Path],
        output_dir: Union[str, Path],
        annotation_file: Union[str, Path],
    ):
        """
        Initialize preprocessor.

        Args:
            video_dir: Directory containing WLASL videos
            output_dir: Directory to save extracted landmarks
            annotation_file: Path to WLASL annotation JSON
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.annotation_file = Path(annotation_file)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_video(self, video_id: str) -> Optional[np.ndarray]:
        """
        Extract landmarks from a single video.

        Args:
            video_id: Video ID

        Returns:
            Landmarks array or None if extraction fails
        """
        try:
            import mediapipe as mp
            import cv2
        except ImportError:
            raise ImportError(
                "MediaPipe and OpenCV required for preprocessing. "
                "Install with: pip install mediapipe opencv-python"
            )

        video_path = self.video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            return None

        # Initialize MediaPipe
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(str(video_path))
        landmarks_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            results = holistic.process(rgb_frame)

            # Extract all landmarks
            frame_landmarks = np.zeros((543, 3), dtype=np.float32)

            # Pose (33 landmarks)
            if results.pose_landmarks:
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    frame_landmarks[i] = [lm.x, lm.y, lm.z]

            # Face (468 landmarks)
            if results.face_landmarks:
                for i, lm in enumerate(results.face_landmarks.landmark):
                    frame_landmarks[33 + i] = [lm.x, lm.y, lm.z]

            # Left hand (21 landmarks)
            if results.left_hand_landmarks:
                for i, lm in enumerate(results.left_hand_landmarks.landmark):
                    frame_landmarks[501 + i] = [lm.x, lm.y, lm.z]

            # Right hand (21 landmarks)
            if results.right_hand_landmarks:
                for i, lm in enumerate(results.right_hand_landmarks.landmark):
                    frame_landmarks[522 + i] = [lm.x, lm.y, lm.z]

            landmarks_list.append(frame_landmarks)

        cap.release()
        holistic.close()

        if not landmarks_list:
            return None

        return np.array(landmarks_list)

    def preprocess_all(
        self,
        subset: str = 'wlasl100',
        verbose: bool = True,
    ) -> Dict[str, int]:
        """
        Preprocess all videos for a subset.

        Args:
            subset: Dataset subset to process
            verbose: Print progress

        Returns:
            Dictionary with processing statistics
        """
        # Load annotations
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)

        subset_num = int(subset.replace('wlasl', ''))

        processed = 0
        failed = 0
        skipped = 0

        for entry in annotations[:subset_num]:
            for instance in entry.get('instances', []):
                video_id = instance.get('video_id', '')
                if not video_id:
                    continue

                output_path = self.output_dir / f"{video_id}.npy"

                # Skip if already processed
                if output_path.exists():
                    skipped += 1
                    continue

                if verbose:
                    print(f"Processing {video_id}...")

                landmarks = self.preprocess_video(video_id)

                if landmarks is not None:
                    np.save(output_path, landmarks)
                    processed += 1
                else:
                    failed += 1

        return {
            'processed': processed,
            'failed': failed,
            'skipped': skipped,
        }
