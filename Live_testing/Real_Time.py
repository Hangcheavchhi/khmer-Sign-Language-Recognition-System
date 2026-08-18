# ============================================================
# LANDMARK EXTRACTION MODULE (For Live Testing)
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SMART ADAPTIVE EXTRACTOR — KEEP (Works for live frames)
# ============================================================
class SmartAdaptiveExtractor:
    def __init__(self):
        self.detection_history = deque(maxlen=10)
        self.hand_size_history = deque(maxlen=10)
        self.frames_since_detection = 0
        self.stats = {
            'normal': 0,
            'adaptive': 0,
            'failed': 0,
            'total': 0,
        }

    def _hand_size(self, results, h, w):
        sizes = []
        for lm_set in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if lm_set:
                xs = [lm.x * w for lm in lm_set.landmark]
                ys = [lm.y * h for lm in lm_set.landmark]
                sizes.append(max(max(xs)-min(xs), max(ys)-min(ys)))
        return np.mean(sizes) if sizes else 0

    def _has_hands(self, results):
        return bool(results.left_hand_landmarks or results.right_hand_landmarks)

    def _scale_back(self, results, scale):
        for lm_set in [results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks]:
            if lm_set:
                for lm in lm_set.landmark:
                    lm.x /= scale
                    lm.y /= scale
        return results

    def extract(self, frame, holistic):
        h, w = frame.shape[:2]
        self.stats['total'] += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        if self._has_hands(results):
            sz = self._hand_size(results, h, w)
            self.hand_size_history.append(sz)
            self.frames_since_detection = 0
            self.stats['normal'] += 1
            return results

        self.frames_since_detection += 1
        avg_sz = np.mean(self.hand_size_history) if self.hand_size_history else 0
        scales = [1.3, 1.6, 2.0] if avg_sz < 80 else [1.3]

        for scale in scales:
            if scale * min(h, w) > 2000:
                continue
            up = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
            res = holistic.process(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
            if self._has_hands(res):
                self._scale_back(res, scale)
                self.stats['adaptive'] += 1
                return res

        self.stats['failed'] += 1
        return results

    def get_stats(self):
        t = max(self.stats['total'], 1)
        return {
            'normal_detection_rate': self.stats['normal'] / t,
            'adaptive_detection_rate': self.stats['adaptive'] / t,
            'failed_detection_rate': self.stats['failed'] / t,
        }


# ============================================================
# FEATURE EXTRACTION FUNCTIONS — KEEP (Work for live frames)
# ============================================================

def extract_pose(pose_lm):
    """Extract 132 pose features"""
    if pose_lm is None:
        return np.zeros(132, dtype=np.float32)
    arr = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_lm.landmark], dtype=np.float32)
    # Normalize relative to hip midpoint
    hip = (arr[23, :3] + arr[24, :3]) / 2
    arr[:, :3] -= hip
    return arr.flatten()

def extract_hand(hand_lm):
    """Extract 63 hand features"""
    if hand_lm is None:
        return np.zeros(63, dtype=np.float32)
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32)
    return (pts - pts[0]).flatten()

def extract_raw_features(results):
    """
    Extract 258 raw features from a single frame
    This is what you'll use in live testing
    """
    pose_f = extract_pose(results.pose_landmarks)   # 132
    left_f = extract_hand(results.left_hand_landmarks)   # 63
    right_f = extract_hand(results.right_hand_landmarks) # 63
    return np.concatenate([pose_f, left_f, right_f])  # 258


# ============================================================
# FRAME BUFFER — NEW (For live testing)
# ============================================================

class FrameBuffer:
    """Collects 30 frames of 258 features for sequence prediction"""
    
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.buffer = []
    
    def add_frame(self, features_258):
        """Add 258 features, return 30-frame sequence when ready"""
        self.buffer.append(features_258)
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)
        
        if len(self.buffer) == self.sequence_length:
            return np.array(self.buffer, dtype=np.float32)  # (30, 258)
        return None
    
    def reset(self):
        self.buffer = []


# ============================================================
# VELOCITY HELPER — NEW (For live testing)
# ============================================================

def add_velocity(sequence_258):
    """
    Add velocity to create 516 features
    EXACT same as your training: vel[1:] = pos[1:] - pos[:-1]
    """
    vel = np.zeros_like(sequence_258)  # (30, 258)
    vel[1:] = sequence_258[1:] - sequence_258[:-1]
    return np.concatenate([sequence_258, vel], axis=1)  # (30, 516)

# ============================================================
# NORMALIZATION MODULE (For Live Testing)
# ============================================================

class SignLanguageNormalizer:
    """
    Normalizer specifically for 516-feature sign language data
    Feature structure: [POSE(132) | HANDS(126) | VELOCITY(258)]
    
    For live testing, use normalize_sequence() with the full buffer
    """
    
    def __init__(self, 
                 use_root_center=True,
                 use_shoulder_width=True,
                 use_hand_normalization=True,
                 use_scale_normalization=True,
                 use_clipping=True,
                 target_hand_size=0.3,
                 clip_bounds=(-2.0, 2.0)):
        """
        Initialize normalizer with configuration
        """
        self.use_root_center = use_root_center
        self.use_shoulder_width = use_shoulder_width
        self.use_hand_normalization = use_hand_normalization
        self.use_scale_normalization = use_scale_normalization
        self.use_clipping = use_clipping
        
        self.target_hand_size = target_hand_size
        self.clip_bounds = clip_bounds
        
        # Feature indices
        self.POSE_START = 0
        self.POSE_END = 132  # 33 landmarks × 4
        self.HANDS_START = 132
        self.HANDS_END = 258  # 126 features
        self.VELOCITY_START = 258
        self.VELOCITY_END = 516  # 258 features
        
        # MediaPipe landmark indices for pose
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        
        # Hand landmark indices (0-20)
        self.WRIST = 0
        self.FINGERTIPS = [4, 8, 12, 16, 20]
        
    def extract_pose_landmarks(self, features):
        """Extract pose landmarks from features. Returns: (33, 4) array"""
        pose = features[self.POSE_START:self.POSE_END].reshape(33, 4)
        return pose
    
    def extract_hand_landmarks(self, features, hand='left'):
        """Extract hand landmarks from features. Returns: (21, 3) array"""
        hand_features = features[self.HANDS_START:self.HANDS_END]
        
        if hand == 'left':
            hand_lms = hand_features[:63].reshape(21, 3)
        else:  # right
            hand_lms = hand_features[63:].reshape(21, 3)
        
        return hand_lms
    
    def extract_velocity(self, features):
        """Extract velocity features"""
        velocity = features[self.VELOCITY_START:self.VELOCITY_END]
        return velocity
    
    def _normalize_hand_scale(self, hand_landmarks):
        """Normalize hand to target size"""
        wrist = hand_landmarks[self.WRIST]
        
        distances = []
        for idx in self.FINGERTIPS:
            if idx < len(hand_landmarks):
                dist = np.linalg.norm(hand_landmarks[idx] - wrist)
                distances.append(dist)
        
        hand_size = np.mean(distances) if distances else 1.0
        
        if hand_size > 0.001:
            scale_factor = self.target_hand_size / hand_size
            hand_landmarks = hand_landmarks * scale_factor
        
        return hand_landmarks
    
    def _compute_velocity(self, pose_hands):
        """Compute velocity from normalized pose+hands"""
        T = pose_hands.shape[0]
        velocity = np.zeros_like(pose_hands)
        
        # Compute velocity as difference between consecutive frames
        velocity[1:] = pose_hands[1:] - pose_hands[:-1]
        # First frame velocity is zero
        
        return velocity
    
    def normalize_sequence(self, features_sequence):
        """
        Normalize a full sequence (30 frames) of 516-feature vectors
        
        Args:
            features_sequence: (T, 516) array where T = sequence_length (30)
        
        Returns:
            normalized_sequence: (T, 516) array with normalized features
        """
        T = features_sequence.shape[0]
        normalized_features = []
        
        # ── First pass: collect shoulder widths ──
        shoulder_widths = []
        for t in range(T):
            features = features_sequence[t]
            pose = self.extract_pose_landmarks(features)
            
            if self.use_shoulder_width:
                left_shoulder = pose[self.LEFT_SHOULDER][:3]
                right_shoulder = pose[self.RIGHT_SHOULDER][:3]
                width = np.linalg.norm(left_shoulder - right_shoulder)
                shoulder_widths.append(width)
            else:
                shoulder_widths.append(1.0)
        
        shoulder_widths = np.array(shoulder_widths)
        
        # ── Second pass: normalize each frame ──
        for t in range(T):
            features = features_sequence[t].copy()
            
            # Extract components
            pose = self.extract_pose_landmarks(features)
            left_hand = self.extract_hand_landmarks(features, 'left')
            right_hand = self.extract_hand_landmarks(features, 'right')
            velocity = self.extract_velocity(features)
            
            # ── 1. Normalize Pose ──
            if self.use_root_center:
                left_hip = pose[self.LEFT_HIP][:3]
                right_hip = pose[self.RIGHT_HIP][:3]
                hip_center = (left_hip + right_hip) / 2
                pose[:, :3] = pose[:, :3] - hip_center
            
            if self.use_shoulder_width:
                width = shoulder_widths[t]
                if width > 0.001:
                    pose[:, :3] = pose[:, :3] / width
                else:
                    pose[:, :3] = 0
            
            # ── 2. Normalize Hands ──
            if self.use_root_center:
                left_hand = left_hand - left_hand[self.WRIST]
                right_hand = right_hand - right_hand[self.WRIST]
            
            if self.use_scale_normalization:
                left_hand = self._normalize_hand_scale(left_hand)
                right_hand = self._normalize_hand_scale(right_hand)
            
            if self.use_hand_normalization:
                left_hand[:, 0] = -left_hand[:, 0]
            
            if self.use_clipping:
                pose[:, :3] = np.clip(pose[:, :3], 
                                     self.clip_bounds[0], 
                                     self.clip_bounds[1])
                left_hand = np.clip(left_hand, 
                                   self.clip_bounds[0], 
                                   self.clip_bounds[1])
                right_hand = np.clip(right_hand, 
                                    self.clip_bounds[0], 
                                    self.clip_bounds[1])
            
            # ── 3. Combine Features ──
            pose_flat = pose.flatten()
            hands_flat = np.concatenate([left_hand.flatten(), right_hand.flatten()])
            combined = np.concatenate([pose_flat, hands_flat])
            combined = np.concatenate([combined, velocity])
            
            normalized_features.append(combined)
        
        normalized_features = np.array(normalized_features)
        
        # ── 4. Recompute Velocity ──
        pose_hands = normalized_features[:, :258]
        velocity = self._compute_velocity(pose_hands)
        normalized_features = np.concatenate([pose_hands, velocity], axis=1)
        
        return normalized_features


# ============================================================
# SINGLE FRAME NORMALIZATION (For live testing before buffer fills)
# ============================================================

def normalize_single_frame(features_516, normalizer):
    """
    Normalize a single frame of 516 features
    This is a simplified version for when you don't have a full sequence yet
    
    Args:
        features_516: (516,) array
        normalizer: SignLanguageNormalizer instance
    
    Returns:
        normalized: (516,) array
    """
    # Convert to sequence format (1, 516)
    sequence = features_516.reshape(1, -1)
    
    # Use the same normalization logic but with dummy velocity
    # Extract components
    pose = normalizer.extract_pose_landmarks(features_516)
    left_hand = normalizer.extract_hand_landmarks(features_516, 'left')
    right_hand = normalizer.extract_hand_landmarks(features_516, 'right')
    velocity = normalizer.extract_velocity(features_516)
    
    # Normalize pose
    if normalizer.use_root_center:
        left_hip = pose[normalizer.LEFT_HIP][:3]
        right_hip = pose[normalizer.RIGHT_HIP][:3]
        hip_center = (left_hip + right_hip) / 2
        pose[:, :3] = pose[:, :3] - hip_center
    
    # Note: shoulder width normalization requires full sequence
    # For single frame, we skip it (will be applied in sequence normalization)
    
    # Normalize hands
    if normalizer.use_root_center:
        left_hand = left_hand - left_hand[normalizer.WRIST]
        right_hand = right_hand - right_hand[normalizer.WRIST]
    
    if normalizer.use_scale_normalization:
        left_hand = normalizer._normalize_hand_scale(left_hand)
        right_hand = normalizer._normalize_hand_scale(right_hand)
    
    if normalizer.use_hand_normalization:
        left_hand[:, 0] = -left_hand[:, 0]
    
    if normalizer.use_clipping:
        pose[:, :3] = np.clip(pose[:, :3], 
                             normalizer.clip_bounds[0], 
                             normalizer.clip_bounds[1])
        left_hand = np.clip(left_hand, 
                           normalizer.clip_bounds[0], 
                           normalizer.clip_bounds[1])
        right_hand = np.clip(right_hand, 
                            normalizer.clip_bounds[0], 
                            normalizer.clip_bounds[1])
    
    # Combine
    pose_flat = pose.flatten()
    hands_flat = np.concatenate([left_hand.flatten(), right_hand.flatten()])
    normalized = np.concatenate([pose_flat, hands_flat, velocity])
    
    return normalized

# ============================================================
# FEATURE ENGINEERING MODULE (For Live Testing)
# ============================================================

class FeatureEngineeringEngine:
    """
    Feature engineering specifically for 516-feature sign language data
    Data structure: [POSE(132) | HANDS(126) | VELOCITY(258)]
    
    For live testing, use process_frame() for single frames
    """
    
    def __init__(self):
        # Feature indices
        self.POSE_START = 0
        self.POSE_END = 132
        self.HANDS_START = 132
        self.HANDS_END = 258
        self.VELOCITY_START = 258
        self.VELOCITY_END = 516
        
        # MediaPipe landmark indices for pose
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.NOSE = 0
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        
        # Hand fingertip indices (0-20)
        self.HAND_INDICES = {
            'wrist': 0,
            'thumb_tip': 4,
            'index_tip': 8,
            'middle_tip': 12,
            'ring_tip': 16,
            'pinky_tip': 20,
            'thumb_base': 1,
            'index_base': 5,
            'middle_base': 9,
            'ring_base': 13,
            'pinky_base': 17
        }
        
        # Joint chains for angle calculation
        self.JOINT_CHAINS = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        # Store feature dimensions for tracking
        self.feature_dims = {}
        
        # For velocity calculation in live testing
        self.prev_features = None
        self.prev_velocity = None
        self.frame_count = 0
    
    # ============================================================
    # EXTRACT COMPONENTS FROM 516 FEATURES
    # ============================================================
    
    def extract_pose_landmarks(self, features):
        """Extract pose landmarks (33, 4) from 516 features"""
        return features[self.POSE_START:self.POSE_END].reshape(33, 4)
    
    def extract_left_hand(self, features):
        """Extract left hand landmarks (21, 3) from 516 features"""
        hand_features = features[self.HANDS_START:self.HANDS_END]
        return hand_features[:63].reshape(21, 3)
    
    def extract_right_hand(self, features):
        """Extract right hand landmarks (21, 3) from 516 features"""
        hand_features = features[self.HANDS_START:self.HANDS_END]
        return hand_features[63:].reshape(21, 3)
    
    def extract_velocity(self, features):
        """Extract velocity (258) from 516 features"""
        return features[self.VELOCITY_START:self.VELOCITY_END]
    
    # ============================================================
    # 1. RELATIVE JOINT COORDINATES
    # ============================================================
    
    def compute_relative_coordinates(self, hand_landmarks):
        """Compute coordinates relative to wrist (21, 3) -> 63 features"""
        wrist = hand_landmarks[0]
        relative = hand_landmarks - wrist
        return relative.flatten()
    
    def compute_finger_relative_positions(self, hand_landmarks):
        """Compute positions of each finger relative to its base"""
        features = []
        for finger_name, indices in self.JOINT_CHAINS.items():
            base = hand_landmarks[indices[0]]
            for idx in indices[1:]:
                relative = hand_landmarks[idx] - base
                features.extend(relative)
        return np.array(features)
    
    # ============================================================
    # 2. HAND-TO-BODY DISTANCES
    # ============================================================
    
    def compute_hand_to_body_distances(self, hand_landmarks, pose_landmarks):
        """Compute distances from hand center to body parts (7 distances)"""
        hand_center = np.mean(hand_landmarks, axis=0)
        
        body_keypoints = {
            'nose': pose_landmarks[self.NOSE][:3],
            'left_shoulder': pose_landmarks[self.LEFT_SHOULDER][:3],
            'right_shoulder': pose_landmarks[self.RIGHT_SHOULDER][:3],
            'left_hip': pose_landmarks[self.LEFT_HIP][:3],
            'right_hip': pose_landmarks[self.RIGHT_HIP][:3],
            'left_elbow': pose_landmarks[self.LEFT_ELBOW][:3],
            'right_elbow': pose_landmarks[self.RIGHT_ELBOW][:3]
        }
        
        distances = []
        for name, pos in body_keypoints.items():
            dist = np.linalg.norm(hand_center - pos)
            distances.append(dist)
        
        return np.array(distances)
    
    def compute_wrist_to_shoulder_ratio(self, hand_landmarks, pose_landmarks, hand_type='left'):
        """Compute ratio of wrist distance to shoulder width (1 feature)"""
        wrist = hand_landmarks[0]
        
        if hand_type == 'left':
            shoulder = pose_landmarks[self.LEFT_SHOULDER][:3]
        else:
            shoulder = pose_landmarks[self.RIGHT_SHOULDER][:3]
        
        wrist_to_shoulder = np.linalg.norm(wrist - shoulder)
        
        shoulder_width = np.linalg.norm(
            pose_landmarks[self.LEFT_SHOULDER][:3] - 
            pose_landmarks[self.RIGHT_SHOULDER][:3]
        )
        
        if shoulder_width > 0.001:
            ratio = wrist_to_shoulder / shoulder_width
        else:
            ratio = 0
        
        return np.array([ratio])
    
    # ============================================================
    # 3. JOINT ANGLES
    # ============================================================
    
    def compute_hand_angles(self, hand_landmarks):
        """Compute angles for all finger joints (15 angles)"""
        angles = []
        
        for finger_name, indices in self.JOINT_CHAINS.items():
            if len(indices) >= 3:
                for i in range(len(indices) - 2):
                    p1 = hand_landmarks[indices[i]]
                    p2 = hand_landmarks[indices[i+1]]
                    p3 = hand_landmarks[indices[i+2]]
                    angle = self._compute_angle(p1, p2, p3)
                    angles.append(angle)
        
        return np.array(angles)
    
    def _compute_angle(self, p1, p2, p3):
        """Compute angle between three points in degrees"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0.001 and v2_norm > 0.001:
            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
        else:
            angle = 0
        
        return angle
    
    def compute_hand_orientation_angles(self, hand_landmarks):
        """Compute overall hand orientation (2 features)"""
        wrist = hand_landmarks[0]
        index_base = hand_landmarks[5]
        pinky_base = hand_landmarks[17]
        
        v1 = index_base - wrist
        v2 = pinky_base - wrist
        
        palm_normal = np.cross(v1, v2)
        if np.linalg.norm(palm_normal) > 0.001:
            palm_normal = palm_normal / np.linalg.norm(palm_normal)
        
        roll = np.arctan2(palm_normal[1], palm_normal[2]) * 180 / np.pi
        pitch = np.arctan2(-palm_normal[0], np.sqrt(palm_normal[1]**2 + palm_normal[2]**2)) * 180 / np.pi
        
        return np.array([roll, pitch])
    
    # ============================================================
    # 4. VELOCITY & ACCELERATION (for live testing)
    # ============================================================
    
    def process_frame(self, features_516):
        """
        Process a single frame of 516 features to 686 engineered features
        For live testing - processes one frame at a time
        
        Args:
            features_516: (516,) array
        
        Returns:
            features_686: (686,) array
        """
        # Extract components
        pose = self.extract_pose_landmarks(features_516)
        left_hand = self.extract_left_hand(features_516)
        right_hand = self.extract_right_hand(features_516)
        velocity = self.extract_velocity(features_516)
        
        # ── Static Features ──
        static_features = []
        
        # 1. Relative coordinates (left: 63, right: 63)
        left_relative = self.compute_relative_coordinates(left_hand)
        right_relative = self.compute_relative_coordinates(right_hand)
        static_features.append(left_relative)
        static_features.append(right_relative)
        
        # 2. Hand-to-body distances (left: 7, right: 7)
        left_distances = self.compute_hand_to_body_distances(left_hand, pose)
        right_distances = self.compute_hand_to_body_distances(right_hand, pose)
        static_features.append(left_distances)
        static_features.append(right_distances)
        
        # 3. Joint angles (left: 15, right: 15)
        left_angles = self.compute_hand_angles(left_hand)
        right_angles = self.compute_hand_angles(right_hand)
        static_features.append(left_angles)
        static_features.append(right_angles)
        
        # 4. Hand orientation (left: 2, right: 2)
        left_orientation = self.compute_hand_orientation_angles(left_hand)
        right_orientation = self.compute_hand_orientation_angles(right_hand)
        static_features.append(left_orientation)
        static_features.append(right_orientation)
        
        # 5. Wrist-to-shoulder ratio (left: 1, right: 1)
        left_ratio = self.compute_wrist_to_shoulder_ratio(left_hand, pose, 'left')
        right_ratio = self.compute_wrist_to_shoulder_ratio(right_hand, pose, 'right')
        static_features.append(left_ratio)
        static_features.append(right_ratio)
        
        # Combine static features (166 features)
        static_combined = np.concatenate(static_features)
        
        # ── Dynamic Features ──
        self.frame_count += 1
        
        # 6. Base features for velocity (pose + hands): 258
        pose_flat = pose.flatten()
        left_flat = left_hand.flatten()
        right_flat = right_hand.flatten()
        base = np.concatenate([pose_flat, left_flat, right_flat])
        
        # 7. Velocity: Use existing velocity (258)
        velocity = self.extract_velocity(features_516)
        
        # 8. Speed (1)
        speed = np.array([np.linalg.norm(velocity)], dtype=np.float32)
        
        # 9. Acceleration (258)
        if self.prev_features is not None:
            acceleration = velocity - self.prev_velocity
        else:
            acceleration = np.zeros(258, dtype=np.float32)
        
        # 10. Acceleration magnitude (1)
        accel_mag = np.array([np.linalg.norm(acceleration)], dtype=np.float32)
        
        # 11. Motion direction (2)
        if np.linalg.norm(velocity) > 0.001:
            direction = velocity / np.linalg.norm(velocity)
            azimuth = np.arctan2(direction[1], direction[0]) * 180 / np.pi
            elevation = np.arctan2(direction[2], np.sqrt(direction[0]**2 + direction[1]**2)) * 180 / np.pi
            direction_vec = np.array([azimuth, elevation], dtype=np.float32)
        else:
            direction_vec = np.zeros(2, dtype=np.float32)
        
        # ── Combine ALL Features (686) ──
        all_features = np.concatenate([
            static_combined,   # 166
            velocity,          # 258
            speed,             # 1
            acceleration,      # 258
            accel_mag,         # 1
            direction_vec      # 2
        ])
        
        # Update previous values
        self.prev_features = base.copy()
        self.prev_velocity = velocity.copy()
        
        return all_features
    
    def process_sequence(self, features_sequence):
        """
        Process a full sequence of 516-feature frames
        For when you have collected all 30 frames
        
        Args:
            features_sequence: (T, 516) array where T = 30
        
        Returns:
            engineered_sequence: (T, 686) array
        """
        T = features_sequence.shape[0]
        engineered_frames = []
        
        # Reset state for sequence processing
        self.prev_features = None
        self.prev_velocity = None
        self.frame_count = 0
        
        for t in range(T):
            engineered = self.process_frame(features_sequence[t])
            engineered_frames.append(engineered)
        
        return np.array(engineered_frames)
    
    def reset(self):
        """Reset the state for a new sequence"""
        self.prev_features = None
        self.prev_velocity = None
        self.frame_count = 0

import os
import sys
import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp
from pathlib import Path
from collections import deque
from enum import Enum
#os.environ['TF_USE_LEGACY_KERAS'] = '1'
# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = r"D:/uni/Intern-1-Project/ksl/models/bgru_best_model_v1.h5"
LABEL_MAP_PATH = r"D:/uni/Intern-1-Project/ksl/models/label_map.json"
SEQ_LEN              = 30
CONF_THRESHOLD       = 0.45
HOLD_FRAMES          = 90
CAMERA_IDX           = 0

# ── Gesture detection ─────────────────────────────────────────
MOVE_START           = 0.003
MOVE_STOP            = 0.001
STILLNESS_FRAMES     = 8
MIN_GESTURE_FRAMES   = 30
MAX_GESTURE_FRAMES   = 180

# ── End detection ─────────────────────────────────────────────
END_DETECTION_FRAMES = 15

# ── Similar sign groups ──────────────────────────────────────
SIMILAR_SIGN_GROUPS = [
    ['ប៊ិក', 'ប៊ិកក្រហម', 'ប៊ិកខៀវ'],
    ['ហ្វឺតក្រហម', 'ហ្វឺតខៀវ', 'ហ្វឺតខ្មៅ'],
]
CAREFUL_CLASSES = set()
for group in SIMILAR_SIGN_GROUPS:
    for c in group:
        CAREFUL_CLASSES.add(c)

# ── Feature config ──────────────────────────────────────────
USE_POSE   = True
USE_HANDS  = True
POSE_FEAT  = 132
HAND_FEAT  = 63
BASE_FEAT  = POSE_FEAT + HAND_FEAT * 2  # 258

# ============================================================
# SMART ADAPTIVE EXTRACTOR
# ============================================================
class SmartAdaptiveExtractor:
    def __init__(self):
        self.detection_history = deque(maxlen=10)
        self.hand_size_history = deque(maxlen=10)
        self.frames_since_detection = 0
        self.stats = {'normal': 0, 'adaptive': 0, 'failed': 0, 'total': 0}

    def _hand_size(self, results, h, w):
        sizes = []
        for lm_set in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if lm_set:
                xs = [lm.x * w for lm in lm_set.landmark]
                ys = [lm.y * h for lm in lm_set.landmark]
                sizes.append(max(max(xs)-min(xs), max(ys)-min(ys)))
        return np.mean(sizes) if sizes else 0

    def _has_hands(self, results):
        return bool(results.left_hand_landmarks or results.right_hand_landmarks)

    def _scale_back(self, results, scale):
        for lm_set in [results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks]:
            if lm_set:
                for lm in lm_set.landmark:
                    lm.x /= scale
                    lm.y /= scale
        return results

    def extract(self, frame, holistic):
        h, w = frame.shape[:2]
        self.stats['total'] += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        if self._has_hands(results):
            sz = self._hand_size(results, h, w)
            self.hand_size_history.append(sz)
            self.frames_since_detection = 0
            self.stats['normal'] += 1
            return results
        self.frames_since_detection += 1
        avg_sz = np.mean(self.hand_size_history) if self.hand_size_history else 0
        scales = [1.3, 1.6, 2.0] if avg_sz < 80 else [1.3]
        for scale in scales:
            if scale * min(h, w) > 2000:
                continue
            up = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
            res = holistic.process(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
            if self._has_hands(res):
                self._scale_back(res, scale)
                self.stats['adaptive'] += 1
                return res
        self.stats['failed'] += 1
        return results

    def get_stats(self):
        t = max(self.stats['total'], 1)
        return {
            'normal_detection_rate': self.stats['normal'] / t,
            'adaptive_detection_rate': self.stats['adaptive'] / t,
            'failed_detection_rate': self.stats['failed'] / t,
        }

# ============================================================
# FRAME BUFFER
# ============================================================
class FrameBuffer:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.buffer = []
    
    def add_frame(self, features):
        self.buffer.append(features)
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)
        if len(self.buffer) == self.sequence_length:
            return np.array(self.buffer, dtype=np.float32)
        return None
    
    def reset(self):
        self.buffer = []

# ============================================================
# NORMALIZER
# ============================================================
class SignLanguageNormalizer:
    def __init__(self, use_root_center=True, use_shoulder_width=True, use_hand_normalization=True,
                 use_scale_normalization=True, use_clipping=True, target_hand_size=0.3, clip_bounds=(-2.0, 2.0)):
        self.use_root_center = use_root_center
        self.use_shoulder_width = use_shoulder_width
        self.use_hand_normalization = use_hand_normalization
        self.use_scale_normalization = use_scale_normalization
        self.use_clipping = use_clipping
        self.target_hand_size = target_hand_size
        self.clip_bounds = clip_bounds
        
        self.POSE_START = 0
        self.POSE_END = 132
        self.HANDS_START = 132
        self.HANDS_END = 258
        self.VELOCITY_START = 258
        self.VELOCITY_END = 516
        
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.WRIST = 0
        self.FINGERTIPS = [4, 8, 12, 16, 20]
    
    def extract_pose_landmarks(self, features):
        return features[self.POSE_START:self.POSE_END].reshape(33, 4)
    
    def extract_hand_landmarks(self, features, hand='left'):
        hand_features = features[self.HANDS_START:self.HANDS_END]
        if hand == 'left':
            return hand_features[:63].reshape(21, 3)
        else:
            return hand_features[63:].reshape(21, 3)
    
    def extract_velocity(self, features):
        return features[self.VELOCITY_START:self.VELOCITY_END]
    
    def _normalize_hand_scale(self, hand_landmarks):
        wrist = hand_landmarks[self.WRIST]
        distances = []
        for idx in self.FINGERTIPS:
            if idx < len(hand_landmarks):
                dist = np.linalg.norm(hand_landmarks[idx] - wrist)
                distances.append(dist)
        hand_size = np.mean(distances) if distances else 1.0
        if hand_size > 0.001:
            scale_factor = self.target_hand_size / hand_size
            hand_landmarks = hand_landmarks * scale_factor
        return hand_landmarks
    
    def _compute_velocity(self, pose_hands):
        T = pose_hands.shape[0]
        velocity = np.zeros_like(pose_hands)
        velocity[1:] = pose_hands[1:] - pose_hands[:-1]
        return velocity
    
    def normalize_sequence(self, features_sequence):
        T = features_sequence.shape[0]
        normalized_features = []
        shoulder_widths = []
        for t in range(T):
            features = features_sequence[t]
            pose = self.extract_pose_landmarks(features)
            if self.use_shoulder_width:
                left_shoulder = pose[self.LEFT_SHOULDER][:3]
                right_shoulder = pose[self.RIGHT_SHOULDER][:3]
                width = np.linalg.norm(left_shoulder - right_shoulder)
                shoulder_widths.append(width)
            else:
                shoulder_widths.append(1.0)
        shoulder_widths = np.array(shoulder_widths)
        for t in range(T):
            features = features_sequence[t].copy()
            pose = self.extract_pose_landmarks(features)
            left_hand = self.extract_hand_landmarks(features, 'left')
            right_hand = self.extract_hand_landmarks(features, 'right')
            velocity = self.extract_velocity(features)
            if self.use_root_center:
                left_hip = pose[self.LEFT_HIP][:3]
                right_hip = pose[self.RIGHT_HIP][:3]
                hip_center = (left_hip + right_hip) / 2
                pose[:, :3] = pose[:, :3] - hip_center
            if self.use_shoulder_width:
                width = shoulder_widths[t]
                if width > 0.001:
                    pose[:, :3] = pose[:, :3] / width
                else:
                    pose[:, :3] = 0
            if self.use_root_center:
                left_hand = left_hand - left_hand[self.WRIST]
                right_hand = right_hand - right_hand[self.WRIST]
            if self.use_scale_normalization:
                left_hand = self._normalize_hand_scale(left_hand)
                right_hand = self._normalize_hand_scale(right_hand)
            if self.use_hand_normalization:
                left_hand[:, 0] = -left_hand[:, 0]
            if self.use_clipping:
                pose[:, :3] = np.clip(pose[:, :3], self.clip_bounds[0], self.clip_bounds[1])
                left_hand = np.clip(left_hand, self.clip_bounds[0], self.clip_bounds[1])
                right_hand = np.clip(right_hand, self.clip_bounds[0], self.clip_bounds[1])
            pose_flat = pose.flatten()
            hands_flat = np.concatenate([left_hand.flatten(), right_hand.flatten()])
            combined = np.concatenate([pose_flat, hands_flat])
            combined = np.concatenate([combined, velocity])
            normalized_features.append(combined)
        normalized_features = np.array(normalized_features)
        pose_hands = normalized_features[:, :258]
        velocity = self._compute_velocity(pose_hands)
        normalized_features = np.concatenate([pose_hands, velocity], axis=1)
        return normalized_features

# ============================================================
# FEATURE ENGINEERING ENGINE
# ============================================================
class FeatureEngineeringEngine:
    def __init__(self):
        self.POSE_START = 0
        self.POSE_END = 132
        self.HANDS_START = 132
        self.HANDS_END = 258
        self.VELOCITY_START = 258
        self.VELOCITY_END = 516
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.NOSE = 0
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.JOINT_CHAINS = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        self.prev_features = None
        self.prev_velocity = None
        self.frame_count = 0
    
    def extract_pose_landmarks(self, features):
        return features[self.POSE_START:self.POSE_END].reshape(33, 4)
    
    def extract_left_hand(self, features):
        hand_features = features[self.HANDS_START:self.HANDS_END]
        return hand_features[:63].reshape(21, 3)
    
    def extract_right_hand(self, features):
        hand_features = features[self.HANDS_START:self.HANDS_END]
        return hand_features[63:].reshape(21, 3)
    
    def extract_velocity(self, features):
        return features[self.VELOCITY_START:self.VELOCITY_END]
    
    def compute_relative_coordinates(self, hand_landmarks):
        wrist = hand_landmarks[0]
        relative = hand_landmarks - wrist
        return relative.flatten()
    
    def compute_hand_to_body_distances(self, hand_landmarks, pose_landmarks):
        hand_center = np.mean(hand_landmarks, axis=0)
        body_keypoints = {
            'nose': pose_landmarks[self.NOSE][:3],
            'left_shoulder': pose_landmarks[self.LEFT_SHOULDER][:3],
            'right_shoulder': pose_landmarks[self.RIGHT_SHOULDER][:3],
            'left_hip': pose_landmarks[self.LEFT_HIP][:3],
            'right_hip': pose_landmarks[self.RIGHT_HIP][:3],
            'left_elbow': pose_landmarks[self.LEFT_ELBOW][:3],
            'right_elbow': pose_landmarks[self.RIGHT_ELBOW][:3]
        }
        distances = []
        for name, pos in body_keypoints.items():
            dist = np.linalg.norm(hand_center - pos)
            distances.append(dist)
        return np.array(distances)
    
    def compute_hand_angles(self, hand_landmarks):
        angles = []
        for finger_name, indices in self.JOINT_CHAINS.items():
            if len(indices) >= 3:
                for i in range(len(indices) - 2):
                    p1 = hand_landmarks[indices[i]]
                    p2 = hand_landmarks[indices[i+1]]
                    p3 = hand_landmarks[indices[i+2]]
                    angle = self._compute_angle(p1, p2, p3)
                    angles.append(angle)
        return np.array(angles)
    
    def _compute_angle(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm > 0.001 and v2_norm > 0.001:
            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
        else:
            angle = 0
        return angle
    
    def compute_hand_orientation_angles(self, hand_landmarks):
        wrist = hand_landmarks[0]
        index_base = hand_landmarks[5]
        pinky_base = hand_landmarks[17]
        v1 = index_base - wrist
        v2 = pinky_base - wrist
        palm_normal = np.cross(v1, v2)
        if np.linalg.norm(palm_normal) > 0.001:
            palm_normal = palm_normal / np.linalg.norm(palm_normal)
        roll = np.arctan2(palm_normal[1], palm_normal[2]) * 180 / np.pi
        pitch = np.arctan2(-palm_normal[0], np.sqrt(palm_normal[1]**2 + palm_normal[2]**2)) * 180 / np.pi
        return np.array([roll, pitch])
    
    def compute_wrist_to_shoulder_ratio(self, hand_landmarks, pose_landmarks, hand_type='left'):
        wrist = hand_landmarks[0]
        if hand_type == 'left':
            shoulder = pose_landmarks[self.LEFT_SHOULDER][:3]
        else:
            shoulder = pose_landmarks[self.RIGHT_SHOULDER][:3]
        wrist_to_shoulder = np.linalg.norm(wrist - shoulder)
        shoulder_width = np.linalg.norm(pose_landmarks[self.LEFT_SHOULDER][:3] - pose_landmarks[self.RIGHT_SHOULDER][:3])
        if shoulder_width > 0.001:
            ratio = wrist_to_shoulder / shoulder_width
        else:
            ratio = 0
        return np.array([ratio])
    
    def process_frame(self, features_516):
        pose = self.extract_pose_landmarks(features_516)
        left_hand = self.extract_left_hand(features_516)
        right_hand = self.extract_right_hand(features_516)
        velocity = self.extract_velocity(features_516)
        static_features = []
        left_relative = self.compute_relative_coordinates(left_hand)
        right_relative = self.compute_relative_coordinates(right_hand)
        static_features.append(left_relative)
        static_features.append(right_relative)
        left_distances = self.compute_hand_to_body_distances(left_hand, pose)
        right_distances = self.compute_hand_to_body_distances(right_hand, pose)
        static_features.append(left_distances)
        static_features.append(right_distances)
        left_angles = self.compute_hand_angles(left_hand)
        right_angles = self.compute_hand_angles(right_hand)
        static_features.append(left_angles)
        static_features.append(right_angles)
        left_orientation = self.compute_hand_orientation_angles(left_hand)
        right_orientation = self.compute_hand_orientation_angles(right_hand)
        static_features.append(left_orientation)
        static_features.append(right_orientation)
        left_ratio = self.compute_wrist_to_shoulder_ratio(left_hand, pose, 'left')
        right_ratio = self.compute_wrist_to_shoulder_ratio(right_hand, pose, 'right')
        static_features.append(left_ratio)
        static_features.append(right_ratio)
        static_combined = np.concatenate(static_features)
        self.frame_count += 1
        pose_flat = pose.flatten()
        left_flat = left_hand.flatten()
        right_flat = right_hand.flatten()
        base = np.concatenate([pose_flat, left_flat, right_flat])
        velocity = self.extract_velocity(features_516)
        speed = np.array([np.linalg.norm(velocity)], dtype=np.float32)
        if self.prev_features is not None:
            acceleration = velocity - self.prev_velocity
        else:
            acceleration = np.zeros(258, dtype=np.float32)
        accel_mag = np.array([np.linalg.norm(acceleration)], dtype=np.float32)
        if np.linalg.norm(velocity) > 0.001:
            direction = velocity / np.linalg.norm(velocity)
            azimuth = np.arctan2(direction[1], direction[0]) * 180 / np.pi
            elevation = np.arctan2(direction[2], np.sqrt(direction[0]**2 + direction[1]**2)) * 180 / np.pi
            direction_vec = np.array([azimuth, elevation], dtype=np.float32)
        else:
            direction_vec = np.zeros(2, dtype=np.float32)
        all_features = np.concatenate([static_combined, velocity, speed, acceleration, accel_mag, direction_vec])
        self.prev_features = base.copy()
        self.prev_velocity = velocity.copy()
        return all_features
    
    def process_sequence(self, features_sequence):
        T = features_sequence.shape[0]
        engineered_frames = []
        self.prev_features = None
        self.prev_velocity = None
        self.frame_count = 0
        for t in range(T):
            engineered = self.process_frame(features_sequence[t])
            engineered_frames.append(engineered)
        return np.array(engineered_frames)
    
    def reset(self):
        self.prev_features = None
        self.prev_velocity = None
        self.frame_count = 0

# ============================================================
# LOAD MODEL & LABELS
# ============================================================
print("Loading model...")
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
print(f" Model loaded! Expects: {model.input_shape}")
MODEL_SEQ = model.input_shape[1]
MODEL_FEATURES = model.input_shape[2]

with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
actions = {int(k): v for k, v in label_map.items()}
print(f" {len(actions)} classes loaded")

# ============================================================
# INITIALIZE COMPONENTS
# ============================================================
extractor = SmartAdaptiveExtractor()
buffer = FrameBuffer(sequence_length=SEQ_LEN)
normalizer = SignLanguageNormalizer()
feature_engine = FeatureEngineeringEngine()

# ============================================================
# MEDIAPIPE
# ============================================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_features(results):
    parts = []
    
    if USE_POSE:
        if results.pose_landmarks:
            arr = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                             for lm in results.pose_landmarks.landmark],
                            dtype=np.float32)
            hip = (arr[23,:3] + arr[24,:3]) / 2
            arr[:,:3] -= hip
            parts.append(arr.flatten())
        else:
            parts.append(np.zeros(POSE_FEAT, dtype=np.float32))
    if USE_HANDS:
        for lm_set in [results.left_hand_landmarks,
                        results.right_hand_landmarks]:
            if lm_set:
                pts = np.array([[lm.x, lm.y, lm.z]
                                  for lm in lm_set.landmark],
                                 dtype=np.float32)
                parts.append((pts - pts[0]).flatten())
            else:
                parts.append(np.zeros(HAND_FEAT, dtype=np.float32))
    return np.concatenate(parts)

def has_hands(results):
    return bool(results.left_hand_landmarks or results.right_hand_landmarks)

# ============================================================
# COLOR DETECTION
# ============================================================
def detect_color_gesture(results):
    if not results.pose_landmarks:
        return 'none'
    if not (results.left_hand_landmarks or results.right_hand_landmarks):
        return 'none'
    
    lms = results.pose_landmarks.landmark
    MOUTH_Y = (lms[9].y + lms[10].y) / 2
    EYEBROW_Y = (lms[1].y + lms[4].y) / 2
    LEFT_CHEEK = lms[7]
    RIGHT_CHEEK = lms[8]
    CHEEK_Y = (LEFT_CHEEK.y + RIGHT_CHEEK.y) / 2
    
    hand_y = None
    hand_landmarks = None
    for lm_set in [results.right_hand_landmarks, results.left_hand_landmarks]:
        if lm_set:
            hand_landmarks = lm_set.landmark
            hand_y = lm_set.landmark[0].y
            break
    
    if hand_y is None:
        return 'none'
    
    # Check BLUE
    if hand_landmarks:
        index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y, hand_landmarks[8].z])
        index_pip = np.array([hand_landmarks[6].x, hand_landmarks[6].y, hand_landmarks[6].z])
        index_mcp = np.array([hand_landmarks[5].x, hand_landmarks[5].y, hand_landmarks[5].z])
        wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z])
        
        tip_to_wrist_dist = np.linalg.norm(index_tip - wrist)
        pip_to_wrist_dist = np.linalg.norm(index_pip - wrist)
        is_extended = tip_to_wrist_dist > pip_to_wrist_dist * 1.2 and tip_to_wrist_dist > 0.08
        is_pointing_up = index_tip[1] < index_pip[1] - 0.02
        
        mcp_to_pip = index_pip - index_mcp
        pip_to_tip = index_tip - index_pip
        if np.linalg.norm(mcp_to_pip) > 0.001:
            mcp_to_pip = mcp_to_pip / np.linalg.norm(mcp_to_pip)
        if np.linalg.norm(pip_to_tip) > 0.001:
            pip_to_tip = pip_to_tip / np.linalg.norm(pip_to_tip)
        dot_product = abs(np.dot(mcp_to_pip, pip_to_tip))
        is_curved = dot_product < 0.85
        
        middle_tip = np.array([hand_landmarks[12].x, hand_landmarks[12].y, hand_landmarks[12].z])
        ring_tip = np.array([hand_landmarks[16].x, hand_landmarks[16].y, hand_landmarks[16].z])
        pinky_tip = np.array([hand_landmarks[20].x, hand_landmarks[20].y, hand_landmarks[20].z])
        middle_extended = np.linalg.norm(middle_tip - wrist) > 0.06
        ring_extended = np.linalg.norm(ring_tip - wrist) > 0.06
        pinky_extended = np.linalg.norm(pinky_tip - wrist) > 0.06
        
        if is_extended and is_curved and is_pointing_up and not middle_extended and not ring_extended and not pinky_extended:
            dist_to_cheek = abs(hand_y - CHEEK_Y)
            dist_to_mouth = abs(hand_y - MOUTH_Y)
            dist_to_eyebrow = abs(hand_y - EYEBROW_Y)
            if dist_to_cheek < 0.15 or dist_to_mouth < 0.15 or dist_to_eyebrow < 0.15:
                return 'blue'
    
    dist_mouth = abs(hand_y - MOUTH_Y)
    if dist_mouth < 0.08:
        return 'red'
    
    dist_eyebrow = abs(hand_y - EYEBROW_Y)
    if dist_eyebrow < 0.08:
        return 'black'
    
    dist_cheek = abs(hand_y - CHEEK_Y)
    if dist_cheek < 0.08 and hand_landmarks:
        index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y, hand_landmarks[8].z])
        index_pip = np.array([hand_landmarks[6].x, hand_landmarks[6].y, hand_landmarks[6].z])
        wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z])
        tip_to_wrist = np.linalg.norm(index_tip - wrist)
        pip_to_wrist = np.linalg.norm(index_pip - wrist)
        if tip_to_wrist > pip_to_wrist * 1.2 and tip_to_wrist > 0.08:
            return 'blue'
    
    return 'none'

# ============================================================
# PROCESS FULL SEQUENCE
# ============================================================
def process_sequence(seq_258):
    vel = np.zeros_like(seq_258)
    vel[1:] = seq_258[1:] - seq_258[:-1]
    seq_516 = np.concatenate([seq_258, vel], axis=1)
    seq_normalized = normalizer.normalize_sequence(seq_516)
    feature_engine.reset()
    seq_686 = feature_engine.process_sequence(seq_normalized)
    return seq_686

# ============================================================
# PREDICT WITH COLOR HISTORY
def predict_with_color_history(gesture_frames_arr, color_history):
    total = len(gesture_frames_arr)
    seq_258 = np.array(gesture_frames_arr, dtype=np.float32)
    seq_686 = process_sequence(seq_258)
    
    if total >= MODEL_SEQ:
        windows = {
            'full': np.linspace(0, MODEL_SEQ-1, MODEL_SEQ).astype(int),
            'last_half': np.linspace(MODEL_SEQ//2, MODEL_SEQ-1, MODEL_SEQ).astype(int),
            'last_third': np.linspace(MODEL_SEQ*2//3, MODEL_SEQ-1, MODEL_SEQ).astype(int),
            'first_half': np.linspace(0, MODEL_SEQ//2, MODEL_SEQ).astype(int),
            'end_only': np.linspace(max(0, MODEL_SEQ - 30), MODEL_SEQ-1, MODEL_SEQ).astype(int),
        }
    else:
        pad_needed = MODEL_SEQ - total
        if pad_needed > 0:
            first_frame = seq_258[0:1].repeat(pad_needed, axis=0)
            seq_258_padded = np.concatenate([first_frame, seq_258], axis=0)
            seq_686 = process_sequence(seq_258_padded)
        windows = {'full': np.arange(MODEL_SEQ)}
    
    preds_dict = {}
    for name, indices in windows.items():
        window_data = seq_686[indices]
        if window_data.shape[1] != MODEL_FEATURES:
            if window_data.shape[1] < MODEL_FEATURES:
                pad = np.zeros((MODEL_SEQ, MODEL_FEATURES - window_data.shape[1]), dtype=np.float32)
                window_data = np.concatenate([window_data, pad], axis=1)
            else:
                window_data = window_data[:, :MODEL_FEATURES]
        inp = np.expand_dims(window_data, axis=0)
        preds_dict[name] = model.predict(inp, verbose=0)[0]
    
    # Get color from history
    detected_color = 'none'
    color_confidence = 0.0
    color_counts = {}
    if color_history:
        for c in color_history:
            if c != 'none':
                color_counts[c] = color_counts.get(c, 0) + 1
        if color_counts:
            max_count = max(color_counts.values())
            detected_color = max(color_counts, key=color_counts.get)
            color_confidence = max_count / len(color_history)
    
    # Get window votes
    window_votes = {}
    window_confidences = {}
    for name, pred in preds_dict.items():
        top_idx = int(np.argmax(pred))
        top_pred = actions.get(top_idx, '?')
        top_conf = float(pred[top_idx])
        window_votes[top_pred] = window_votes.get(top_pred, 0) + 1
        if top_pred not in window_confidences:
            window_confidences[top_pred] = []
        window_confidences[top_pred].append(top_conf)
    
    top_pred = max(window_votes.items(), key=lambda x: x[1])[0]
    base_class = None
    is_pen = False
    is_marker = False
    if 'ហ្វឺត' in top_pred:
        base_class = 'ហ្វឺត'
        is_marker = True
    elif 'ប៊ិក' in top_pred:
        base_class = 'ប៊ិក'
        is_pen = True
    else:
        base_class = top_pred
    
    final_pred = top_pred
    final_conf = np.mean(window_confidences[top_pred]) if top_pred in window_confidences else 0.5
    color_map = {'red': 'ក្រហម', 'black': 'ខ្មៅ', 'blue': 'ខៀវ'}
    color_suffix = color_map.get(detected_color, '')
    
    if detected_color != 'none' and color_suffix and base_class in ['ហ្វឺត', 'ប៊ិក']:
        color_variant = f"{base_class}{color_suffix}"
        for idx, name in actions.items():
            if name == color_variant:
                final_pred = name
                if color_variant in window_confidences:
                    final_conf = np.mean(window_confidences[color_variant])
                else:
                    final_conf = max(0.5, color_confidence)
                break
    
    final_idx = 0
    for idx, name in actions.items():
        if name == final_pred:
            final_idx = idx
            break
    
    if preds_dict:
        avg_pred = np.mean(list(preds_dict.values()), axis=0)
    else:
        avg_pred = np.zeros(len(actions))
        avg_pred[final_idx] = 1.0
    
    return final_idx, final_conf, False, avg_pred

# ============================================================
# STATE MACHINE
# ============================================================
class GestureState(Enum):
    IDLE       = "Waiting..."
    COLLECTING = "Recording..."
    PREDICTING = "Analyzing..."
    SHOWING    = "Done"

state = GestureState.IDLE
gesture_frames = []
prev_features = None
movement = 0.0
stillness_counter = 0
movement_history = deque(maxlen=5)
end_frames_count = 0
current_label = ""
current_conf = 0.0
prediction_hold = 0
sign_evolved_flag = False
color_gesture_history = deque(maxlen=60)

cap = cv2.VideoCapture(CAMERA_IDX)
if not cap.isOpened():
    for idx in [1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened(): break
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow('KSL Live', cv2.WINDOW_NORMAL)
cv2.resizeWindow('KSL Live', 1280, 720)
print("\n Webcam ready! Press Q to quit")
print(" Color Detection Guide:")
print("   - RED: Point hand to MOUTH ")
print("   - BLACK: Point hand to EYEBROW ")
print("   - BLUE: Point hand to CHEEK/EYE ")
print("-" * 50)

# ============================================================
# MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(CAMERA_IDX)
if not cap.isOpened():
    for idx in [1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened(): 
            break

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n Webcam ready! Press Q to quit")
print("-" * 50)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: 
        continue
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    
    # ── Draw landmarks only ──
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(60,60,60), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(60,60,60), thickness=1)
        )
    for lm_set, color in [
        (results.left_hand_landmarks, (0, 200, 0)),
        (results.right_hand_landmarks, (0, 200, 200))
    ]:
        if lm_set:
            mp_drawing.draw_landmarks(
                frame, lm_set,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=color, thickness=2)
            )
    
    # ── Extract features ──
    features = extract_features(results)
    hands_on = has_hands(results)
    
    # ── Compute movement ──
    if prev_features is not None:
        hand_curr = features[POSE_FEAT:]
        hand_prev = prev_features[POSE_FEAT:]
        raw_move = float(np.mean(np.abs(hand_curr - hand_prev)))
    else:
        raw_move = 0.0
    prev_features = features.copy()
    movement_history.append(raw_move)
    movement = float(np.mean(movement_history))
    
    # ── State Machine ──
    if state == GestureState.IDLE:
        if hands_on and movement > MOVE_START:
            state = GestureState.COLLECTING
            gesture_frames = [features.copy()]
            stillness_counter = 0
            end_frames_count = 0
            sign_evolved_flag = False
            color_gesture_history.clear()
            feature_engine.reset()
            
    elif state == GestureState.COLLECTING:
        gesture_frames.append(features.copy())
        n = len(gesture_frames)
        
        # Track color gesture
        color = detect_color_gesture(results)
        color_gesture_history.append(color)
        
        # End detection
        sign_ending = False
        
        if not hands_on:
            end_frames_count += 1
            if end_frames_count >= END_DETECTION_FRAMES and n >= MIN_GESTURE_FRAMES:
                sign_ending = True
        else:
            end_frames_count = 0
        
        if movement < MOVE_STOP:
            stillness_counter += 1
            if stillness_counter >= STILLNESS_FRAMES and n >= MIN_GESTURE_FRAMES:
                sign_ending = True
        else:
            stillness_counter = 0
        
        if n >= MAX_GESTURE_FRAMES:
            sign_ending = True
        
        if sign_ending:
            state = GestureState.PREDICTING
    
    elif state == GestureState.PREDICTING:
        seq_arr = np.array(gesture_frames, dtype=np.float32)
        
        final_idx, final_conf, evolved, avg_pred = predict_with_color_history(
            seq_arr, color_gesture_history
        )
        
        sign_evolved_flag = evolved
        predicted_name = actions.get(final_idx, "Unknown")
        is_careful = predicted_name in CAREFUL_CLASSES
        threshold = 0.40 if is_careful else CONF_THRESHOLD
        
        color_gesture_history.clear()
        
        # ── Print prediction to terminal ──
        if final_conf >= threshold:
            current_label = predicted_name
            current_conf = final_conf
            prediction_hold = HOLD_FRAMES
            print(f"\n{'='*60}")
            print(f" PREDICTION")
            print(f"{'='*60}")
            print(f"   Sign: {current_label}")
            print(f"   Confidence: {final_conf*100:.1f}%")
            print(f"{'='*60}\n")
        else:
            top2 = np.argsort(avg_pred)[::-1][:2]
            current_label = f"{actions.get(top2[0],'?')} / {actions.get(top2[1],'?')}?"
            current_conf = final_conf
            prediction_hold = 45
            print(f"\n{'='*60}")
            print(f" LOW CONFIDENCE")
            print(f"{'='*60}")
            print(f"   Best: {actions.get(top2[0],'?')} ({final_conf*100:.1f}%)")
            print(f"   Alternative: {actions.get(top2[1],'?')}")
            print(f"{'='*60}\n")
        
        state = GestureState.SHOWING
        gesture_frames = []
        stillness_counter = 0
        end_frames_count = 0
        feature_engine.reset()
    
    elif state == GestureState.SHOWING:
        prediction_hold -= 1
        if prediction_hold <= 0:
            state = GestureState.IDLE
            current_label = ""
            current_conf = 0.0
            sign_evolved_flag = False
    
    # ── Show frame with landmarks ──
    cv2.imshow('KSL Live', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
holistic.close()
cv2.destroyAllWindows()
print("\n Done!")


