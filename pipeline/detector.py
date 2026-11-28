"""
MediaPipe face mesh + pose wrapper.
Returns expression label and gesture label each frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe solutions
# ---------------------------------------------------------------------------
_mp_face = mp.solutions.face_mesh
_mp_pose = mp.solutions.pose
_mp_drawing = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class DetectionResult:
    expression: str = "neutral"       # happy / surprised / sad / angry / disgusted / fearful / neutral
    gesture: Optional[str] = None     # hug / pointing / None
    face_detected: bool = False
    face_landmarks: Optional[object] = None
    pose_landmarks: Optional[object] = None


# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe Face Mesh 468-pt model)
# ---------------------------------------------------------------------------
# Mouth corners
_LEFT_MOUTH  = 61
_RIGHT_MOUTH = 291
# Upper / lower lip mid-points
_UPPER_LIP   = 13
_LOWER_LIP   = 14
# Left eye
_LEFT_EYE_TOP    = 159
_LEFT_EYE_BOTTOM = 145
_LEFT_EYE_LEFT   = 33
_LEFT_EYE_RIGHT  = 133
# Right eye
_RIGHT_EYE_TOP    = 386
_RIGHT_EYE_BOTTOM = 374
_RIGHT_EYE_LEFT   = 362
_RIGHT_EYE_RIGHT  = 263
# Eyebrow inner points
_LEFT_BROW_INNER  = 55
_RIGHT_BROW_INNER = 285
# Nose tip
_NOSE_TIP = 1
# Chin
_CHIN = 152
# Face width reference (cheek bones)
_LEFT_CHEEK  = 234
_RIGHT_CHEEK = 454


def _dist(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _face_width(lms) -> float:
    return max(_dist(lms[_LEFT_CHEEK], lms[_RIGHT_CHEEK]), 1e-6)


# ---------------------------------------------------------------------------
# Expression classifier
# ---------------------------------------------------------------------------
def _classify_expression(face_lms) -> str:
    """
    Heuristic expression classification from face mesh landmarks.
    Returns one of: happy, surprised, sad, angry, disgusted, fearful, neutral.
    """
    lms = face_lms.landmark
    fw = _face_width(lms)

    # --- Mouth openness (vertical) ---
    mouth_open = _dist(lms[_UPPER_LIP], lms[_LOWER_LIP]) / fw

    # --- Smile: mouth corners lifted relative to lip centre ---
    mouth_centre_y = (lms[_UPPER_LIP].y + lms[_LOWER_LIP].y) / 2
    left_corner_lift  = mouth_centre_y - lms[_LEFT_MOUTH].y
    right_corner_lift = mouth_centre_y - lms[_RIGHT_MOUTH].y
    smile_ratio = (left_corner_lift + right_corner_lift) / 2 / fw

    # --- Eye openness ---
    left_eye_open  = _dist(lms[_LEFT_EYE_TOP],  lms[_LEFT_EYE_BOTTOM])  / fw
    right_eye_open = _dist(lms[_RIGHT_EYE_TOP], lms[_RIGHT_EYE_BOTTOM]) / fw
    eye_open_avg = (left_eye_open + right_eye_open) / 2

    # --- Eyebrow raise: inner brows vs nose tip ---
    brow_raise = (_dist(lms[_LEFT_BROW_INNER],  lms[_NOSE_TIP]) +
                  _dist(lms[_RIGHT_BROW_INNER], lms[_NOSE_TIP])) / 2 / fw

    # --- Frown: brows pulled together (small inner-brow horizontal distance) ---
    brow_furrow = _dist(lms[_LEFT_BROW_INNER], lms[_RIGHT_BROW_INNER]) / fw

    # Thresholds derived empirically from MediaPipe landmark geometry
    is_smiling   = smile_ratio > 0.018
    big_smile    = smile_ratio > 0.030
    mouth_open_t = mouth_open > 0.045
    wide_open    = mouth_open > 0.075
    eyes_wide    = eye_open_avg > 0.038
    brow_raised  = brow_raise > 0.42
    brow_frowning = brow_furrow < 0.055

    # Decision tree (order matters: most specific first)
    if big_smile and wide_open:
        return "happy"
    if big_smile:
        return "happy"
    if brow_raised and wide_open and eyes_wide:
        return "surprised"
    if brow_raised and mouth_open_t:
        return "surprised"
    if brow_frowning and mouth_open_t:
        return "angry"
    if brow_frowning and not is_smiling:
        # distinguish angry vs sad: sad has downturned corners
        down_corners = -smile_ratio > 0.010
        if down_corners:
            return "sad"
        return "angry"
    if is_smiling:
        return "happy"
    if eyes_wide and not mouth_open_t:
        return "fearful"
    if mouth_open_t and not brow_raised:
        return "disgusted"
    return "neutral"


# ---------------------------------------------------------------------------
# Pose gesture classifier
# ---------------------------------------------------------------------------
# MediaPipe Pose landmark indices
_P = mp.solutions.pose.PoseLandmark

def _classify_gesture(pose_lms, image_w: int, image_h: int) -> Optional[str]:
    """
    Classify body gesture from pose landmarks.
    Returns 'hug', 'pointing', or None.
    """
    lms = pose_lms.landmark

    def vis(idx) -> float:
        return lms[idx].visibility

    def pos(idx):
        return np.array([lms[idx].x, lms[idx].y])

    # Require reasonable visibility for key joints
    required = [_P.LEFT_WRIST, _P.RIGHT_WRIST,
                _P.LEFT_ELBOW, _P.RIGHT_ELBOW,
                _P.LEFT_SHOULDER, _P.RIGHT_SHOULDER]
    if any(vis(j) < 0.4 for j in required):
        return None

    l_wrist    = pos(_P.LEFT_WRIST)
    r_wrist    = pos(_P.RIGHT_WRIST)
    l_elbow    = pos(_P.LEFT_ELBOW)
    r_elbow    = pos(_P.RIGHT_ELBOW)
    l_shoulder = pos(_P.LEFT_SHOULDER)
    r_shoulder = pos(_P.RIGHT_SHOULDER)

    shoulder_mid = (l_shoulder + r_shoulder) / 2
    shoulder_w = abs(l_shoulder[0] - r_shoulder[0])
    if shoulder_w < 1e-6:
        return None

    # --- HUG detection ---
    # Both wrists are close together and near torso midline / crossed
    wrist_dist   = np.linalg.norm(l_wrist - r_wrist) / shoulder_w
    # Both wrists should be near torso (y close to shoulder midpoint y, x near centre)
    l_wrist_torso_x = abs(l_wrist[0] - shoulder_mid[0]) / shoulder_w
    r_wrist_torso_x = abs(r_wrist[0] - shoulder_mid[0]) / shoulder_w
    wrists_cross = l_wrist[0] > r_wrist[0]  # left wrist right of right wrist = arms crossed

    if (wrist_dist < 1.2
            and l_wrist_torso_x < 0.9
            and r_wrist_torso_x < 0.9
            and wrists_cross):
        return "hug"

    # --- POINTING detection ---
    # One arm extended: wrist significantly above shoulder height (smaller y) and
    # elbow angle near 180° (arm straight)
    for wrist, elbow, shoulder in [
        (l_wrist, l_elbow, l_shoulder),
        (r_wrist, r_elbow, r_shoulder),
    ]:
        # Wrist above shoulder (y axis is inverted in image coords)
        wrist_above_shoulder = wrist[1] < shoulder[1] - 0.05
        # Elbow angle
        vec_se = elbow    - shoulder
        vec_ew = wrist    - elbow
        cos_angle = np.dot(vec_se, vec_ew) / (
            np.linalg.norm(vec_se) * np.linalg.norm(vec_ew) + 1e-6
        )
        arm_straight = cos_angle > 0.75  # ~40° or less from straight

        if wrist_above_shoulder and arm_straight:
            return "pointing"

    return None


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------
class KindnessDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._face_mesh = _mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._pose = _mp_pose.Pose(
            model_complexity=0,          # fastest model
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, bgr_frame: np.ndarray) -> DetectionResult:
        """Run face mesh + pose on a BGR frame, return DetectionResult."""
        h, w = bgr_frame.shape[:2]
        rgb = bgr_frame[:, :, ::-1].copy()
        rgb.flags.writeable = False

        face_result = self._face_mesh.process(rgb)
        pose_result = self._pose.process(rgb)

        result = DetectionResult()

        if face_result.multi_face_landmarks:
            result.face_detected = True
            result.face_landmarks = face_result.multi_face_landmarks[0]
            result.expression = _classify_expression(result.face_landmarks)

        if pose_result.pose_landmarks:
            result.pose_landmarks = pose_result.pose_landmarks
            g = _classify_gesture(pose_result.pose_landmarks, w, h)
            if g:
                result.gesture = g

        return result

    def draw_landmarks(self, bgr_frame: np.ndarray, det: DetectionResult) -> np.ndarray:
        """Draw face mesh and pose landmarks onto the frame (in-place) and return it."""
        if det.face_landmarks:
            _mp_drawing.draw_landmarks(
                bgr_frame,
                det.face_landmarks,
                _mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=_mp_drawing_styles.get_default_face_mesh_contours_style(),
            )
        if det.pose_landmarks:
            _mp_drawing.draw_landmarks(
                bgr_frame,
                det.pose_landmarks,
                _mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=_mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        return bgr_frame

    def close(self):
        self._face_mesh.close()
        self._pose.close()
