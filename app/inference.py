import os
import sys
import base64
import cv2
import cv2.aruco as aruco
import torch
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy import stats as scipy_stats
import joblib
import json
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models



# Setup paths to import packages
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(workspace_dir, 'packages'))

from depth_anything_v2_metric.dpt import DepthAnythingV2
from ultralytics import YOLO

# Global constants
INTRINSICS = (2123, 2123, 1500, 2000)
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Marker physical size in meters (48.75 mm == 5 cm printed, with border)
ARUCO_MARKER_SIZE_M = 0.04875

# DepthAnythingV2 configs
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}
encoder = 'vits'
dataset = 'hypersim'
max_depth = 5


def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decodes image bytes to an RGB numpy array.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def detect_aruco(img_rgb: np.ndarray):
    """
    Attempts to detect a 6x6 ArUco marker (ID 0) in the image using multiple
    flip strategies as a robust fallback. Returns:
        found      (bool)          - Whether marker ID 0 was detected
        corners    (np.ndarray)    - Corner array of marker ID 0 in original coords (4, 2)
        flip_type  (str)           - One of 'none', 'horizontal', 'vertical', 'both'
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    flip_variants = [
        ('none',       None),
        ('horizontal', 1),
        ('vertical',   0),
        ('both',       -1),
    ]

    for flip_type, flip_code in flip_variants:
        img_f = img_bgr if flip_code is None else cv2.flip(img_bgr, flip_code)
        corners, ids, _ = detector.detectMarkers(img_f)

        if ids is not None and 0 in ids.flatten():
            idx_0 = list(ids.flatten()).index(0)
            c0 = corners[idx_0][0].copy()  # (4, 2) in flipped coords

            # Map corners back to original image coordinates
            if flip_type == 'horizontal':
                c0[:, 0] = w - c0[:, 0]
            elif flip_type == 'vertical':
                c0[:, 1] = h - c0[:, 1]
            elif flip_type == 'both':
                c0[:, 0] = w - c0[:, 0]
                c0[:, 1] = h - c0[:, 1]

            return True, c0, flip_type

    return False, None, 'none'


def _compute_aruco_scale(corners_orig: np.ndarray):
    """
    Computes the pixel-to-metric scale factor from the detected ArUco marker corners.
    Returns (aruco_scale, z_marker):
        aruco_scale  - meters per pixel
        z_marker     - estimated distance to marker in meters (using mean focal length)
    """
    fx, fy, _, _ = INTRINSICS
    f = (fx + fy) / 2.0

    sides = [
        np.linalg.norm(corners_orig[i] - corners_orig[(i + 1) % 4])
        for i in range(4)
    ]
    s_pixel = np.mean(sides)
    aruco_scale = ARUCO_MARKER_SIZE_M / s_pixel  # meters per pixel
    z_marker = f * aruco_scale
    return aruco_scale, z_marker


class MangoPredictor:
    def __init__(self):
        self.seg_model = None
        self.depth_model = None

        self.orientation_cnn = None
        self.class_mapping = []
        self.morph_l1_models = {}
        self.morph_features = {}
        self.ripe_models = {}
        self.ripe_features = {}
        self.seed_model = None
        self.scaler_l2 = None
        self.seed_features = []

        # --- ArUco-calibrated (aruco) model set ---
        self.aruco_morph_l1_models = {}
        self.aruco_morph_features = {}
        self.aruco_ripe_models = {}
        self.aruco_ripe_features = {}
        self.aruco_seed_model = None
        self.aruco_scaler_l2 = None
        self.aruco_seed_features = []

        self.orientations = ['side', 'top', 'bottom', 'front', 'back']

    def load_models(self):
        print(f"Loading models on device: {DEVICE}...")

        # 1. YOLO Segmentation Model (shared)
        seg_path = os.path.join(workspace_dir, "models/YoloV11n/runs/segment/mango_seg_v1-4/weights/best.pt")
        print(f"Loading YOLO from {seg_path}...")
        self.seg_model = YOLO(seg_path)

        # 2. Depth Anything V2 Model (shared)
        depth_path = os.path.join(workspace_dir, f"models/DepthAnythingV2/depth_anything_v2_metric_{dataset}_{encoder}.pth")
        print(f"Loading DepthAnythingV2 from {depth_path}...")
        self.depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        self.depth_model.load_state_dict(torch.load(depth_path, map_location='cpu'))
        self.depth_model = self.depth_model.to(DEVICE).eval()

        # Load unified CNN Orientation classifier
        print("Loading unified CNN orientation model...")
        cnn_path = os.path.join(workspace_dir, "models/orientation/orientation_cnn.pth")
        mapping_path = os.path.join(workspace_dir, "models/orientation/class_mapping.json")
        
        with open(mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
            
        self.orientation_cnn = models.mobilenet_v3_small()
        num_features = self.orientation_cnn.classifier[3].in_features
        self.orientation_cnn.classifier[3] = torch.nn.Linear(num_features, len(self.class_mapping))
        self.orientation_cnn.load_state_dict(torch.load(cnn_path, map_location='cpu'))
        self.orientation_cnn = self.orientation_cnn.to(DEVICE).eval()

        # ============================================================
        # Load UNCALIBRATED (aruco_no) models
        # ============================================================
        print("Loading uncalibrated (aruco_no) model set...")
        for orient in self.orientations:
            morph_dir = os.path.join(workspace_dir, f'models/aruco_no/morphology_external/{orient}')
            ripe_dir = os.path.join(workspace_dir, f'models/aruco_no/ripeness/{orient}')
            self.morph_l1_models[orient] = (
                joblib.load(os.path.join(morph_dir, 'morphology_model.joblib')),
                joblib.load(os.path.join(morph_dir, 'scaler_morph.joblib'))
            )
            with open(os.path.join(morph_dir, 'features.json'), 'r') as f:
                self.morph_features[orient] = json.load(f)
            self.ripe_models[orient] = (
                joblib.load(os.path.join(ripe_dir, 'ripeness_model.joblib')),
                joblib.load(os.path.join(ripe_dir, 'scaler_ripe.joblib'))
            )
            with open(os.path.join(ripe_dir, 'features.json'), 'r') as f:
                self.ripe_features[orient] = json.load(f)

        seed_dir_no = os.path.join(workspace_dir, 'models/aruco_no/morphology_internal')
        self.seed_model = joblib.load(os.path.join(seed_dir_no, 'seed_model.joblib'))
        self.scaler_l2 = joblib.load(os.path.join(seed_dir_no, 'scaler_l2.joblib'))
        with open(os.path.join(seed_dir_no, 'features.json'), 'r') as f:
            self.seed_features = json.load(f)

        # ============================================================
        # Load ARUCO-CALIBRATED (aruco) models
        # ============================================================
        print("Loading ArUco-calibrated model set...")
        for orient in self.orientations:
            morph_dir = os.path.join(workspace_dir, f'models/aruco/morphology_external/{orient}')
            ripe_dir = os.path.join(workspace_dir, f'models/aruco/ripeness/{orient}')
            self.aruco_morph_l1_models[orient] = (
                joblib.load(os.path.join(morph_dir, 'morphology_model.joblib')),
                joblib.load(os.path.join(morph_dir, 'scaler_morph.joblib'))
            )
            with open(os.path.join(morph_dir, 'features.json'), 'r') as f:
                self.aruco_morph_features[orient] = json.load(f)
            self.aruco_ripe_models[orient] = (
                joblib.load(os.path.join(ripe_dir, 'ripeness_model.joblib')),
                joblib.load(os.path.join(ripe_dir, 'scaler_ripe.joblib'))
            )
            with open(os.path.join(ripe_dir, 'features.json'), 'r') as f:
                self.aruco_ripe_features[orient] = json.load(f)

        seed_dir_aruco = os.path.join(workspace_dir, 'models/aruco/morphology_internal')
        self.aruco_seed_model = joblib.load(os.path.join(seed_dir_aruco, 'seed_model.joblib'))
        self.aruco_scaler_l2 = joblib.load(os.path.join(seed_dir_aruco, 'scaler_l2.joblib'))
        with open(os.path.join(seed_dir_aruco, 'features.json'), 'r') as f:
            self.aruco_seed_features = json.load(f)

        print("All models (uncalibrated + ArUco-calibrated) successfully loaded!")

    # ------------------------------------------------------------------
    # Feature extraction — UNCALIBRATED (monocular depth only)
    # ------------------------------------------------------------------
    def extract_features(self, raw_img: np.ndarray, intrinsics=INTRINSICS):
        """
        Extracts 3D features from a single image using monocular depth only (no ArUco).
        Returns (features_dict, refined_mask, depth_map) or (None, None, None).
        """
        fx, fy, cx, cy = intrinsics

        seg_results_temp = self.seg_model(raw_img, verbose=False)[0]
        depth_map = self.depth_model.infer_image(raw_img)

        if len(seg_results_temp.boxes) == 0:
            return None, None, None

        img_h, img_w = seg_results_temp.orig_shape
        img_center = np.array([img_w / 2, img_h / 2])
        box_centers = seg_results_temp.boxes.xywh[:, 0:2].cpu().numpy()
        distances = np.linalg.norm(box_centers - img_center, axis=1)
        closest_idx = distances.argmin()
        selected_obj = seg_results_temp[closest_idx]

        if selected_obj.masks is None:
            return None, None, None

        yolo_mask = selected_obj.masks.data[0].cpu().numpy().astype(np.uint8)
        yolo_mask = cv2.resize(yolo_mask, (raw_img.shape[1], raw_img.shape[0]))

        mango_depths = depth_map[yolo_mask > 0]
        if len(mango_depths) == 0:
            return None, None, None

        z_min, z_max = np.percentile(mango_depths, [0, 96])
        refined_mask = (yolo_mask > 0) & (depth_map >= z_min) & (depth_map <= z_max)

        v, u = np.where(refined_mask)
        z = depth_map[refined_mask]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_3d = np.stack((x, y, z), axis=-1)

        min_bound, max_bound = points_3d.min(axis=0), points_3d.max(axis=0)
        dims = max_bound - min_bound
        W, H, T = dims[0], dims[1], dims[2]

        ratio_lw = H / W if W != 0 else 0
        ratio_tl = T / H if H != 0 else 0
        ratio_tw = T / W if W != 0 else 0

        geometric_mean_dim = (W * H * T) ** (1 / 3)
        longest_axis = max(W, H, T)
        sphericity = geometric_mean_dim / longest_axis if longest_axis != 0 else 0

        hull = ConvexHull(points_3d)
        vol_est = hull.volume
        solidity_3d = vol_est / (W * H * T) if (W * H * T) != 0 else 0

        dz_dv, dz_du = np.gradient(depth_map)
        curvature_val = np.sqrt(dz_du[refined_mask] ** 2 + dz_dv[refined_mask] ** 2)
        mean_curvature = np.mean(curvature_val)
        std_curvature = np.std(curvature_val)

        lab = cv2.cvtColor(raw_img, cv2.COLOR_RGB2Lab)
        avg_a = np.mean(lab[:, :, 1][refined_mask])
        avg_b = np.mean(lab[:, :, 2][refined_mask])

        # Group B: Depth distribution statistics
        z_full = depth_map[refined_mask]
        d_p4  = float(np.percentile(z_full, 4))
        d_p25 = float(np.percentile(z_full, 25))
        d_p75 = float(np.percentile(z_full, 75))
        d_p96 = float(np.percentile(z_full, 96))
        depth_range = d_p96 - d_p4
        depth_iqr   = d_p75 - d_p25
        depth_skew  = float(scipy_stats.skew(z_full))
        depth_kurt  = float(scipy_stats.kurtosis(z_full))
        lap = cv2.Laplacian(depth_map.astype(np.float32), cv2.CV_32F)
        depth_lap_std = float(np.std(lap[refined_mask]))

        # Group A: 2D mask contour shape features
        mask_u8 = refined_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox_w_px = float(np.max(u) - np.min(u))
        bbox_h_px = float(np.max(v) - np.min(v))
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            mask_area_px_val = float(cv2.contourArea(cnt))
            perimeter_px = float(cv2.arcLength(cnt, True))
            mask_2d_roundness = float((4 * np.pi * mask_area_px_val) / (perimeter_px ** 2)) if perimeter_px > 0 else 0.0
            hull_2d = cv2.convexHull(cnt)
            hull_area_2d = float(cv2.contourArea(hull_2d))
            mask_2d_solidity = mask_area_px_val / hull_area_2d if hull_area_2d > 0 else 0.0
            if len(cnt) >= 5:
                (ex, ey), (eminor, emajor), eangle = cv2.fitEllipse(cnt)
                ellipse_major_px = float(emajor)
                ellipse_minor_px = float(eminor)
                ellipse_aspect_ratio = float(eminor / emajor) if emajor > 0 else 0.0
                ellipse_angle_deg = float(eangle)
            else:
                ellipse_major_px = bbox_h_px
                ellipse_minor_px = bbox_w_px
                ellipse_aspect_ratio = bbox_w_px / bbox_h_px if bbox_h_px > 0 else 0.0
                ellipse_angle_deg = 0.0
        else:
            mask_area_px_val = float(np.sum(refined_mask))
            perimeter_px = 0.0
            mask_2d_roundness = 0.0
            mask_2d_solidity = 0.0
            ellipse_major_px = bbox_h_px
            ellipse_minor_px = bbox_w_px
            ellipse_aspect_ratio = bbox_w_px / bbox_h_px if bbox_h_px > 0 else 0.0
            ellipse_angle_deg = 0.0

        # Group C: Extended color features (CIELab + HSV)
        avg_l = float(np.mean(lab[:, :, 0][refined_mask]))
        avg_a = float(np.mean(lab[:, :, 1][refined_mask]))
        avg_b_lab = float(np.mean(lab[:, :, 2][refined_mask]))
        std_a = float(np.std(lab[:, :, 1][refined_mask]))
        std_b = float(np.std(lab[:, :, 2][refined_mask]))
        hsv = cv2.cvtColor(raw_img, cv2.COLOR_RGB2HSV)
        color_hue_mean = float(np.mean(hsv[:, :, 0][refined_mask]))
        color_hue_std  = float(np.std(hsv[:, :, 0][refined_mask]))
        color_sat_mean = float(np.mean(hsv[:, :, 1][refined_mask]))

        distance_z = float(np.median(z))
        mango_area_px = float(np.sum(refined_mask))

        features = {
            'width': W, 'height': H, 'thickness': T,
            'sphericity': sphericity,
            'volume_hull': vol_est,
            'surface_area_hull': hull.area / 2,
            'ratio_lw': ratio_lw,
            'ratio_tl': ratio_tl,
            'ratio_tw': ratio_tw,
            'solidity_3d': solidity_3d,
            'mean_curvature': mean_curvature,
            'curvature_dev': std_curvature,
            'bbox_w_px': bbox_w_px,
            'bbox_h_px': bbox_h_px,
            'distance_z': distance_z,
            'color_a': avg_a,
            'color_b': avg_b_lab,
            # Group A
            'mask_area_px': mango_area_px,
            'mask_perimeter_px': perimeter_px,
            'mask_2d_roundness': mask_2d_roundness,
            'mask_2d_solidity': mask_2d_solidity,
            'ellipse_major_px': ellipse_major_px,
            'ellipse_minor_px': ellipse_minor_px,
            'ellipse_aspect_ratio': ellipse_aspect_ratio,
            'ellipse_angle_deg': ellipse_angle_deg,
            # Group B
            'depth_range': depth_range,
            'depth_iqr': depth_iqr,
            'depth_p25': d_p25,
            'depth_p75': d_p75,
            'depth_skewness': depth_skew,
            'depth_kurtosis': depth_kurt,
            'depth_lap_std': depth_lap_std,
            # Group C
            'color_l': avg_l,
            'color_std_a': std_a,
            'color_std_b': std_b,
            'color_hue_mean': color_hue_mean,
            'color_hue_std': color_hue_std,
            'color_saturation': color_sat_mean,
        }

        # Engineer features on features dict if needed (to ensure shape alignment)
        features_df = pd.DataFrame([features])
        from notebook_aruco_run import engineer_features
        features_df = engineer_features(features_df)
        features = features_df.iloc[0].to_dict()

        return features, refined_mask.astype(np.uint8), depth_map

    # ------------------------------------------------------------------
    # Feature extraction — ARUCO-CALIBRATED
    # ------------------------------------------------------------------
    def extract_features_aruco(self, raw_img: np.ndarray, aruco_corners: np.ndarray, intrinsics=INTRINSICS):
        """
        Extracts an extended feature set calibrated via the ArUco marker.
        Ports the logic from notebook_aruco_run.py::extract_mango_features().
        Returns (features_dict, refined_mask, calibrated_depth_map) or (None, None, None).
        """
        fx, fy, cx, cy = intrinsics
        h, w = raw_img.shape[:2]

        # Compute ArUco scale and calibrated distance
        aruco_scale, z_marker = _compute_aruco_scale(aruco_corners)

        # YOLO Segmentation
        seg_results_temp = self.seg_model(raw_img, verbose=False)[0]
        if len(seg_results_temp.boxes) == 0:
            return None, None, None

        img_center = np.array([w / 2, h / 2])
        box_centers = seg_results_temp.boxes.xywh[:, 0:2].cpu().numpy()
        distances = np.linalg.norm(box_centers - img_center, axis=1)
        closest_idx = distances.argmin()
        selected_obj = seg_results_temp[closest_idx]

        if selected_obj.masks is None:
            return None, None, None

        yolo_mask = selected_obj.masks.data[0].cpu().numpy().astype(np.uint8)
        yolo_mask = cv2.resize(yolo_mask, (w, h))

        # Depth map + ArUco calibration
        depth_map = self.depth_model.infer_image(raw_img)

        aruco_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(aruco_mask, [aruco_corners.astype(np.int32)], 255)

        depth_aruco = depth_map[aruco_mask > 0]
        if len(depth_aruco) == 0:
            # Fallback: sample around marker centre
            mc = np.mean(aruco_corners, axis=0).astype(np.int32)
            depth_aruco = depth_map[
                max(0, mc[1] - 10):min(h, mc[1] + 10),
                max(0, mc[0] - 10):min(w, mc[0] + 10)
            ].ravel()

        d_marker = np.median(depth_aruco)
        depth_scale = z_marker / d_marker if d_marker > 0 else 1.0
        cal_depth = depth_map * depth_scale

        # Depth noise reduction
        mango_depths = cal_depth[yolo_mask > 0]
        if len(mango_depths) == 0:
            return None, None, None

        z_min, z_max = np.percentile(mango_depths, [0, 96])
        refined_mask = (yolo_mask > 0) & (cal_depth >= z_min) & (cal_depth <= z_max)

        # 3D projection
        v, u = np.where(refined_mask)
        z = cal_depth[refined_mask]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_3d = np.stack((x, y, z), axis=-1)

        min_bound, max_bound = points_3d.min(axis=0), points_3d.max(axis=0)
        dims = max_bound - min_bound
        W, H, T = dims[0], dims[1], dims[2]

        ratio_lw = H / W if W != 0 else 0
        ratio_tl = T / H if H != 0 else 0
        ratio_tw = T / W if W != 0 else 0

        geometric_mean_dim = (W * H * T) ** (1 / 3)
        longest_axis = max(W, H, T)
        sphericity = geometric_mean_dim / longest_axis if longest_axis != 0 else 0

        hull = ConvexHull(points_3d)
        vol_est = hull.volume
        solidity_3d = vol_est / (W * H * T) if (W * H * T) != 0 else 0

        dz_dv, dz_du = np.gradient(cal_depth)
        curvature_val = np.sqrt(dz_du[refined_mask] ** 2 + dz_dv[refined_mask] ** 2)
        mean_curvature = float(np.mean(curvature_val))
        std_curvature = float(np.std(curvature_val))

        # Group B: Depth distribution statistics
        z_full = cal_depth[refined_mask]
        d_p4  = float(np.percentile(z_full, 4))
        d_p25 = float(np.percentile(z_full, 25))
        d_p75 = float(np.percentile(z_full, 75))
        d_p96 = float(np.percentile(z_full, 96))
        depth_range = d_p96 - d_p4
        depth_iqr   = d_p75 - d_p25
        depth_skew  = float(scipy_stats.skew(z_full))
        depth_kurt  = float(scipy_stats.kurtosis(z_full))
        lap = cv2.Laplacian(cal_depth.astype(np.float32), cv2.CV_32F)
        depth_lap_std = float(np.std(lap[refined_mask]))

        # Group A: 2D mask contour shape features
        mask_u8 = refined_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox_w_px = float(np.max(u) - np.min(u))
        bbox_h_px = float(np.max(v) - np.min(v))
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            mask_area_px_val = float(cv2.contourArea(cnt))
            perimeter_px = float(cv2.arcLength(cnt, True))
            mask_2d_roundness = float((4 * np.pi * mask_area_px_val) / (perimeter_px ** 2)) if perimeter_px > 0 else 0.0
            hull_2d = cv2.convexHull(cnt)
            hull_area_2d = float(cv2.contourArea(hull_2d))
            mask_2d_solidity = mask_area_px_val / hull_area_2d if hull_area_2d > 0 else 0.0
            if len(cnt) >= 5:
                (ex, ey), (eminor, emajor), eangle = cv2.fitEllipse(cnt)
                ellipse_major_px = float(emajor)
                ellipse_minor_px = float(eminor)
                ellipse_aspect_ratio = float(eminor / emajor) if emajor > 0 else 0.0
                ellipse_angle_deg = float(eangle)
            else:
                ellipse_major_px = bbox_h_px
                ellipse_minor_px = bbox_w_px
                ellipse_aspect_ratio = bbox_w_px / bbox_h_px if bbox_h_px > 0 else 0.0
                ellipse_angle_deg = 0.0
        else:
            mask_area_px_val = float(np.sum(refined_mask))
            perimeter_px = 0.0
            mask_2d_roundness = 0.0
            mask_2d_solidity = 0.0
            ellipse_major_px = bbox_h_px
            ellipse_minor_px = bbox_w_px
            ellipse_aspect_ratio = bbox_w_px / bbox_h_px if bbox_h_px > 0 else 0.0
            ellipse_angle_deg = 0.0

        # Group C: Extended colour features (CIELab + HSV)
        lab = cv2.cvtColor(raw_img, cv2.COLOR_RGB2Lab)
        avg_l = float(np.mean(lab[:, :, 0][refined_mask]))
        avg_a = float(np.mean(lab[:, :, 1][refined_mask]))
        avg_b_lab = float(np.mean(lab[:, :, 2][refined_mask]))
        std_a = float(np.std(lab[:, :, 1][refined_mask]))
        std_b = float(np.std(lab[:, :, 2][refined_mask]))
        hsv = cv2.cvtColor(raw_img, cv2.COLOR_RGB2HSV)
        color_hue_mean = float(np.mean(hsv[:, :, 0][refined_mask]))
        color_hue_std  = float(np.std(hsv[:, :, 0][refined_mask]))
        color_sat_mean = float(np.mean(hsv[:, :, 1][refined_mask]))

        distance_z = float(np.median(z))

        # ArUco-specific engineered metrics
        mango_w_metric_aruco = bbox_w_px * aruco_scale
        mango_h_metric_aruco = bbox_h_px * aruco_scale
        mango_area_px = float(np.sum(refined_mask))
        mango_area_metric_aruco = mango_area_px * (aruco_scale ** 2)

        features = {
            # Core 3D / ratio features
            'width': W, 'height': H, 'thickness': T,
            'sphericity': sphericity,
            'volume_hull': vol_est,
            'surface_area_hull': hull.area / 2,
            'ratio_lw': ratio_lw,
            'ratio_tl': ratio_tl,
            'ratio_tw': ratio_tw,
            'solidity_3d': solidity_3d,
            'mean_curvature': mean_curvature,
            'curvature_dev': std_curvature,
            # Bounding box & distance
            'bbox_w_px': bbox_w_px,
            'bbox_h_px': bbox_h_px,
            'distance_z': distance_z,
            # Original colour
            'color_a': avg_a,
            'color_b': avg_b_lab,
            # Group A: 2D mask shape
            'mask_area_px': mango_area_px,
            'mask_perimeter_px': perimeter_px,
            'mask_2d_roundness': mask_2d_roundness,
            'mask_2d_solidity': mask_2d_solidity,
            'ellipse_major_px': ellipse_major_px,
            'ellipse_minor_px': ellipse_minor_px,
            'ellipse_aspect_ratio': ellipse_aspect_ratio,
            'ellipse_angle_deg': ellipse_angle_deg,
            # Group B: Depth distribution
            'depth_range': depth_range,
            'depth_iqr': depth_iqr,
            'depth_p25': d_p25,
            'depth_p75': d_p75,
            'depth_skewness': depth_skew,
            'depth_kurtosis': depth_kurt,
            'depth_lap_std': depth_lap_std,
            # Group C: Extended colour & texture
            'color_l': avg_l,
            'color_std_a': std_a,
            'color_std_b': std_b,
            'color_hue_mean': color_hue_mean,
            'color_hue_std': color_hue_std,
            'color_saturation': color_sat_mean,
            # ArUco-specific engineered metrics
            'aruco_scale': float(aruco_scale),
            'aruco_distance': float(z_marker),
            'mango_w_metric_aruco': float(mango_w_metric_aruco),
            'mango_h_metric_aruco': float(mango_h_metric_aruco),
            'mango_area_px': mango_area_px,
            'mango_area_metric_aruco': float(mango_area_metric_aruco),
        }

        return features, refined_mask.astype(np.uint8), cal_depth

    # ------------------------------------------------------------------
    # End-to-end prediction (auto-selects pipeline based on ArUco detection)
    # ------------------------------------------------------------------
    def predict(self, raw_img: np.ndarray, intrinsics=INTRINSICS):
        """
        Executes the end-to-end inference pipeline.
        Automatically selects the ArUco-calibrated path if a marker is detected,
        otherwise falls back to the monocular-depth-only path.
        """
        # --- Step 1: ArUco Detection ---
        aruco_found, aruco_corners, _ = detect_aruco(raw_img)

        # --- Step 2: Feature Extraction ---
        if aruco_found:
            extract_res = self.extract_features_aruco(raw_img, aruco_corners, intrinsics)
        else:
            extract_res = self.extract_features(raw_img, intrinsics)

        if extract_res is None or extract_res[0] is None:
            return {"success": False, "error": "No mango detected in the image or failed to extract features."}

        features, mask, depth_map = extract_res

        # --- Step 3: Select model set ---
        if aruco_found:
            morph_models = self.aruco_morph_l1_models
            morph_feats_map = self.aruco_morph_features
            ripe_models = self.aruco_ripe_models
            ripe_feats_map = self.aruco_ripe_features
            seed_model = self.aruco_seed_model
            scaler_l2 = self.aruco_scaler_l2
            seed_feats = self.aruco_seed_features
        else:
            morph_models = self.morph_l1_models
            morph_feats_map = self.morph_features
            ripe_models = self.ripe_models
            ripe_feats_map = self.ripe_features
            seed_model = self.seed_model
            scaler_l2 = self.scaler_l2
            seed_feats = self.seed_features

        # --- Step 4: CNN Orientation Prediction ---
        v, u = np.where(mask > 0)
        if len(v) > 0:
            y1, y2 = np.min(v), np.max(v)
            x1, x2 = np.min(u), np.max(u)
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)
            x1 = max(0, x1 - pad_x)
            x2 = min(raw_img.shape[1], x2 + pad_x)
            y1 = max(0, y1 - pad_y)
            y2 = min(raw_img.shape[0], y2 + pad_y)
            crop = raw_img[y1:y2, x1:x2]
        else:
            crop = raw_img

        crop_resized = cv2.resize(crop, (224, 224))
        crop_pil = Image.fromarray(crop_resized)
        
        cnn_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = cnn_transform(crop_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.orientation_cnn(input_tensor)
            _, predicted_idx = outputs.max(1)
            predicted_orient = self.class_mapping[predicted_idx.item()]

        if predicted_orient not in morph_models:
            predicted_orient = 'side'

        # --- Step 5: Layer 1 — Morphology & Ripeness ---
        morph_l1_model, scaler_l1 = morph_models[predicted_orient]
        ripe_model, scaler_ripe = ripe_models[predicted_orient]

        morph_features_list = morph_feats_map[predicted_orient]
        ripe_features_list  = ripe_feats_map[predicted_orient]

        input_morph_df = pd.DataFrame([features], columns=morph_features_list)
        input_ripe_df  = pd.DataFrame([features], columns=ripe_features_list)

        morph_scaled = scaler_l1.transform(input_morph_df)
        ripe_scaled  = scaler_ripe.transform(input_ripe_df)

        morph_scaled_df = pd.DataFrame(morph_scaled, columns=morph_features_list)
        ripe_scaled_df  = pd.DataFrame(ripe_scaled, columns=ripe_features_list)

        external_preds = morph_l1_model.predict(morph_scaled_df)[0]
        ripe_pred = ripe_model.predict(ripe_scaled_df)[0]
        is_ripe_string = "Ripe" if ripe_pred == 1 else "Unripe"

        # --- Step 6: Layer 2 — Seed (Endocarp) Prediction ---
        l2_numerical_features = ['m_height', 'm_width', 'm_thickness', 'm_volume', 'm_weight']
        input_l2_numerical = pd.DataFrame([external_preds], columns=l2_numerical_features)

        # Get expected features for this scaler to avoid ValueError
        scaler_expected_features = list(scaler_l2.feature_names_in_)
        input_l2_scaled = pd.DataFrame(
            scaler_l2.transform(input_l2_numerical[scaler_expected_features]),
            columns=scaler_expected_features
        )
        input_l2_scaled['is_ripe'] = int(ripe_pred)



        # Get the expected features for the seed model estimator
        if hasattr(seed_model, 'feature_names_in_'):
            model_expected_features = list(seed_model.feature_names_in_)
        elif len(seed_model.estimators_) > 0 and hasattr(seed_model.estimators_[0], 'feature_names_in_'):
            model_expected_features = list(seed_model.estimators_[0].feature_names_in_)
        else:
            model_expected_features = seed_feats

        input_l2_df = input_l2_scaled[model_expected_features]
        internal_preds = seed_model.predict(input_l2_df)[0]

        # --- Step 7: Visualization overlays ---
        mask_overlay = raw_img.copy()
        green_mask = np.zeros_like(raw_img)
        green_mask[:, :] = [46, 204, 113]
        mask_indices = mask > 0
        mask_overlay[mask_indices] = (
            raw_img[mask_indices] * 0.4 + green_mask[mask_indices] * 0.6
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_overlay, contours, -1, (46, 204, 113), 3)

        # If ArUco was found, draw the marker outline on the mask overlay
        if aruco_found and aruco_corners is not None:
            pts = aruco_corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(mask_overlay, [pts], isClosed=True, color=(90, 200, 255), thickness=3)
            center = tuple(np.mean(aruco_corners, axis=0).astype(int))
            cv2.putText(mask_overlay, "ArUco", (center[0] - 28, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 200, 255), 2)

        # Depth map visualization
        depth_map_viz = depth_map.copy()
        depth_min = depth_map[mask_indices].min()
        depth_max = depth_map[mask_indices].max()
        depth_map_viz[~mask_indices] = depth_max

        if depth_max - depth_min > 0:
            norm_depth = (depth_map_viz - depth_min) / (depth_max - depth_min)
            norm_depth = 1.0 - norm_depth
            norm_depth = (np.clip(norm_depth, 0, 1) * 255.0).astype(np.uint8)
        else:
            norm_depth = np.zeros_like(depth_map_viz, dtype=np.uint8)

        norm_depth[~mask_indices] = 0
        depth_colormap = cv2.applyColorMap(norm_depth, cv2.COLORMAP_MAGMA)
        depth_colormap[~mask_indices] = [18, 18, 24]
        depth_colormap_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

        # Encode to base64
        original_b64 = self.encode_base64(raw_img)
        mask_b64 = self.encode_base64(mask_overlay)
        depth_b64 = self.encode_base64(depth_colormap_rgb)

        return {
            "success": True,
            "aruco_detected": bool(aruco_found),
            "pipeline_mode": "aruco_calibrated" if aruco_found else "monocular_only",
            "predictions": {
                "orientation": predicted_orient,
                "is_ripe": is_ripe_string,
                "ripeness_raw": int(ripe_pred),
                "mango": {
                    "height_cm": float(external_preds[0]),
                    "width_cm":  float(external_preds[1]),
                    "thickness_cm": float(external_preds[2]),
                    "volume_cm3": float(external_preds[3]),
                    "weight_g":  float(external_preds[4])
                },
                "seed": {
                    "height_cm": float(internal_preds[0]),
                    "width_cm":  float(internal_preds[1]),
                    "thickness_cm": float(internal_preds[2]),
                    "volume_cm3": float(internal_preds[3]),
                    "weight_g":  float(internal_preds[4])
                },
                "distance_z_m":    float(features['distance_z']),
                "volume_hull_m3":  float(features['volume_hull']),
                "sphericity":      float(features['sphericity'])
            },
            "visualizations": {
                "original": original_b64,
                "mask": mask_b64,
                "depth": depth_b64
            }
        }

    @staticmethod
    def encode_base64(img_rgb: np.ndarray) -> str:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', img_bgr)
        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
