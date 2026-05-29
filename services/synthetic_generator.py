"""
synthetic_generator.py
======================
Core engine: 3D rendering → compositing → augmentation → YOLO export.
"""

from __future__ import annotations

import os
import random
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrender
import trimesh
import yaml
from werkzeug.datastructures import FileStorage

from services.synthetic_config import (
    AugmentationConfig,
    LightingConfig,
    PlacementZone,
    ProductClass,
    RotationConfig,
    SyntheticConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deg2rad(d: float) -> float:
    return d * np.pi / 180.0


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Build 4x4 homogeneous rotation matrix from Euler angles (degrees)."""
    cx, sx = np.cos(_deg2rad(rx)), np.sin(_deg2rad(rx))
    cy, sy = np.cos(_deg2rad(ry)), np.sin(_deg2rad(ry))
    cz, sz = np.cos(_deg2rad(rz)), np.sin(_deg2rad(rz))
    Rx = np.array([[1,0,0,0],[0,cx,-sx,0],[0,sx,cx,0],[0,0,0,1]], dtype=np.float64)
    Ry = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]], dtype=np.float64)
    Rz = np.array([[cz,-sz,0,0],[sz,cz,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _point_in_polygon(x: float, y: float, polygon: List[Tuple[int, int]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    px, py = x, y
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _compute_iou(box1: Tuple[int,int,int,int], box2: Tuple[int,int,int,int]) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area1 = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area2 = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# ProductRenderer — wraps trimesh + pyrender
# ---------------------------------------------------------------------------

class ProductRenderer:
    """Renders a 3D model to an RGBA numpy array at a given rotation."""

    def __init__(self, product: ProductClass, lighting: LightingConfig):
        self._product = product
        self._lighting = lighting
        self._renderer = None
        self._current_size: Optional[int] = None
        self._lock = threading.Lock()

    def _ensure_renderer(self, size: int) -> None:
        if self._renderer is None or self._current_size != size:
            if self._renderer is not None:
                try:
                    self._renderer.delete()
                except Exception:
                    pass
            self._renderer = pyrender.OffscreenRenderer(size, size)
            self._current_size = size

    def render(self, rx: float, ry: float, rz: float, render_size: int) -> np.ndarray:
        """
        Returns an RGBA image (H, W, 4) with transparent background.
        Renders the 3D model at the given Euler angles (degrees).
        """
        valid_models = [
            p for p in self._product.model_paths
            if p.lower().endswith(('.obj', '.glb', '.gltf', '.stl', '.ply', '.dae'))
        ]
        if not valid_models:
            raise RuntimeError(f"No valid 3D models found for product {self._product.class_name}")
        model_path = random.choice(valid_models)

        with self._lock:
            try:
                loaded = trimesh.load(model_path)
            except Exception as exc:
                raise RuntimeError(f"Failed to load 3D model {model_path}: {exc}") from exc

            # Normalize: center and scale to unit box
            if hasattr(loaded, "bounds") and loaded.bounds is not None:
                extents = loaded.extents
                max_extent = extents.max() if extents.max() > 0 else 1.0
                T = np.eye(4)
                T[:3, 3] = -loaded.centroid
                loaded.apply_transform(T)
                S = np.eye(4)
                S[0, 0] = S[1, 1] = S[2, 2] = 1.0 / max_extent
                loaded.apply_transform(S)

            # Apply rotation
            R = _rotation_matrix(rx, ry, rz)
            loaded.apply_transform(R)

            # Randomize light intensity
            rng_vary = self._lighting.intensity_variation
            light_intensity = self._lighting.directional_intensity * (
                1.0 + random.uniform(-rng_vary, rng_vary)
            )
            ambient = [
                self._lighting.ambient_intensity,
                self._lighting.ambient_intensity,
                self._lighting.ambient_intensity,
            ]

            if isinstance(loaded, trimesh.Scene):
                scene = pyrender.Scene.from_trimesh_scene(
                    loaded, 
                    bg_color=[0.0, 0.0, 0.0, 0.0], 
                    ambient_light=ambient
                )
            else:
                scene = pyrender.Scene(
                    bg_color=[0.0, 0.0, 0.0, 0.0],
                    ambient_light=ambient,
                )
                mesh_node = pyrender.Mesh.from_trimesh(loaded)
                scene.add(mesh_node)

            # Camera positioned along +Z axis looking at origin
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
            camera_pose = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 2.5],
                [0, 0, 0, 1],
            ], dtype=np.float64)
            scene.add(camera, pose=camera_pose)

            # Directional light from above-front with random angle variation
            angle_offset = random.uniform(*self._lighting.directional_angle_range)
            light_pose = _rotation_matrix(angle_offset, 0, 0) @ camera_pose
            light = pyrender.DirectionalLight(
                color=np.ones(3, dtype=np.float32),
                intensity=float(light_intensity),
            )
            scene.add(light, pose=light_pose)

            # Second fill light from the opposite side
            fill_pose = _rotation_matrix(-angle_offset * 0.5, 180, 0) @ camera_pose
            fill_light = pyrender.DirectionalLight(
                color=np.ones(3, dtype=np.float32),
                intensity=float(light_intensity * 0.4),
            )
            scene.add(fill_light, pose=fill_pose)

            self._ensure_renderer(render_size)
            color, _ = self._renderer.render(
                scene, flags=pyrender.RenderFlags.RGBA
            )
            return color  # (H, W, 4) uint8

    def close(self) -> None:
        with self._lock:
            if self._renderer is not None:
                try:
                    self._renderer.delete()
                except Exception:
                    pass
                self._renderer = None


# ---------------------------------------------------------------------------
# ImageAugmentor
# ---------------------------------------------------------------------------

class ImageAugmentor:
    """Applies realistic augmentations to a composited BGR image."""

    def __init__(self, cfg: AugmentationConfig):
        self._cfg = cfg

    def augment(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()

        if random.random() < self._cfg.brightness_prob:
            img = self._brightness(img)

        if random.random() < self._cfg.hue_shift_prob:
            img = self._hue_shift(img)

        if random.random() < self._cfg.noise_prob:
            img = self._gaussian_noise(img)

        if random.random() < self._cfg.motion_blur_prob:
            img = self._motion_blur(img)

        if random.random() < self._cfg.jpeg_artifact_prob:
            img = self._jpeg_artifact(img)

        return img

    def _brightness(self, img: np.ndarray) -> np.ndarray:
        lo, hi = self._cfg.brightness_range
        factor = random.uniform(lo, hi)
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def _hue_shift(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
        shift = random.randint(*self._cfg.hue_shift_range)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        lo, hi = self._cfg.noise_std_range
        std = random.uniform(lo, hi)
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _motion_blur(self, img: np.ndarray) -> np.ndarray:
        lo, hi = self._cfg.motion_blur_kernel_range
        k = random.randrange(lo, hi + 1, 2)
        if k < 3:
            k = 3
        angle = random.uniform(0, 180)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0
        kernel /= k
        M = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        kernel /= kernel.sum() if kernel.sum() > 0 else 1
        return cv2.filter2D(img, -1, kernel)

    def _jpeg_artifact(self, img: np.ndarray) -> np.ndarray:
        lo, hi = self._cfg.jpeg_quality_range
        quality = random.randint(lo, hi)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img


# ---------------------------------------------------------------------------
# SceneCompositor
# ---------------------------------------------------------------------------

class SceneCompositor:
    """Composites an RGBA product render onto a BGR background."""

    def __init__(self, config: SyntheticConfig):
        self._config = config

    def composite(
        self,
        background: np.ndarray,
        product_rgba: np.ndarray,
        center_x: int,
        center_y: int,
        scale: float,
        feather_px: int,
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Places the product centered at (center_x, center_y) on background.
        Returns (composited_image, (x1,y1,x2,y2)) or (background, None) if invisible.
        """
        h_bg, w_bg = background.shape[:2]
        h_r, w_r = product_rgba.shape[:2]

        # Scale
        new_w = max(1, int(w_r * scale))
        new_h = max(1, int(h_r * scale))
        scaled = cv2.resize(product_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Position (top-left corner)
        x1 = center_x - new_w // 2
        y1 = center_y - new_h // 2
        x2 = x1 + new_w
        y2 = y1 + new_h

        # Compute intersection with background
        sx1 = max(x1, 0)
        sy1 = max(y1, 0)
        sx2 = min(x2, w_bg)
        sy2 = min(y2, h_bg)

        if sx2 <= sx1 or sy2 <= sy1:
            return background.copy(), None

        # Crop regions
        px1 = sx1 - x1
        py1 = sy1 - y1
        px2 = px1 + (sx2 - sx1)
        py2 = py1 + (sy2 - sy1)

        product_crop = scaled[py1:py2, px1:px2]
        alpha = product_crop[:, :, 3].astype(np.float32) / 255.0

        # Optional edge feathering
        if feather_px > 0:
            k = feather_px * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)

        result = background.copy()
        bg_region = result[sy1:sy2, sx1:sx2].astype(np.float32)
        fg_region = product_crop[:, :, :3].astype(np.float32)

        # Color Harmonization
        if self._config.augmentation.harmonize_color:
            strength = self._config.augmentation.harmonize_strength
            # Calculate mean color of the background region
            bg_mean = cv2.mean(bg_region, mask=(alpha * 255).astype(np.uint8))[:3]
            fg_mean = cv2.mean(fg_region, mask=(alpha * 255).astype(np.uint8))[:3]
            
            # Avoid division by zero
            if sum(fg_mean) > 0 and sum(bg_mean) > 0:
                # Calculate color shift ratios
                shift_ratio = [bg_mean[i] / max(1.0, fg_mean[i]) for i in range(3)]
                
                # Apply shift to foreground with strength blending
                for c in range(3):
                    fg_region[:, :, c] = fg_region[:, :, c] * (1.0 - strength + shift_ratio[c] * strength)
                
                # Re-clip values to valid image range
                fg_region = np.clip(fg_region, 0, 255)

        alpha_3 = alpha[:, :, np.newaxis]
        blended = fg_region * alpha_3 + bg_region * (1.0 - alpha_3)
        result[sy1:sy2, sx1:sx2] = np.clip(blended, 0, 255).astype(np.uint8)

        # Compute tight bounding box from alpha mask in background space
        alpha_full = np.zeros((h_bg, w_bg), dtype=np.uint8)
        alpha_full[sy1:sy2, sx1:sx2] = (alpha * 255).astype(np.uint8)
        ys, xs = np.where(alpha_full > 10)
        if len(xs) == 0:
            return result, None

        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        return result, bbox


# ---------------------------------------------------------------------------
# SyntheticDatasetGenerator — orchestrator
# ---------------------------------------------------------------------------

class GenerationResult:
    def __init__(self):
        self.total = 0
        self.generated = 0
        self.failed = 0
        self.skipped = 0
        self.dataset_id: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.errors: List[str] = []


class SyntheticDatasetGenerator:
    """
    Orchestrates the full pipeline:
    3D render → composite → augment → save → register in TrainingStore.
    """

    STOP_SENTINEL = object()

    def __init__(self, config: SyntheticConfig, store=None):
        self._config = config
        self._store = store
        self._stop_event = threading.Event()
        self._renderers: Dict[int, ProductRenderer] = {}

    def stop(self) -> None:
        self._stop_event.set()

    def _get_renderer(self, product: ProductClass) -> ProductRenderer:
        if product.class_id not in self._renderers:
            self._renderers[product.class_id] = ProductRenderer(product, self._config.lighting)
        return self._renderers[product.class_id]

    def _close_renderers(self) -> None:
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:
                pass
        self._renderers.clear()

    # --- Sampling helpers ---

    def _sample_product(self) -> ProductClass:
        weights = [p.weight for p in self._config.products]
        return random.choices(self._config.products, weights=weights, k=1)[0]

    def _sample_position(self, zone: PlacementZone) -> Tuple[int, int]:
        """Sample a random point inside a polygon zone."""
        pts = zone.points
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        for _ in range(200):
            x = random.randint(min_x, max_x)
            y = random.randint(min_y, max_y)
            if _point_in_polygon(x, y, pts):
                return x, y
        # Fallback: centroid
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

    def _check_overlap(
        self,
        new_box: Tuple[int, int, int, int],
        placed_boxes: List[Tuple[int, int, int, int]],
    ) -> bool:
        """Returns True if new_box overlaps too much with any placed box."""
        for box in placed_boxes:
            if _compute_iou(new_box, box) > self._config.iou_overlap_limit:
                return True
        return False

    def _check_visible(
        self,
        bbox: Tuple[int, int, int, int],
        product_rgba: np.ndarray,
        scale: float,
    ) -> bool:
        """Check if enough of the product is actually visible (not clipped)."""
        # Find the true tight bounding box of the rendered object
        ys, xs = np.where(product_rgba[:, :, 3] > 10)
        if len(xs) == 0:
            return False
            
        true_w = (xs.max() - xs.min()) * scale
        true_h = (ys.max() - ys.min()) * scale
        true_area = max(1.0, true_w * true_h)
        
        # Bounding box on the background
        bx1, by1, bx2, by2 = bbox
        visible_area = (bx2 - bx1) * (by2 - by1)
        
        return (visible_area / true_area) >= self._config.min_visible_fraction

    # --- YOLO label helpers ---

    @staticmethod
    def _bbox_to_yolo(
        bbox: Tuple[int, int, int, int],
        image_w: int,
        image_h: int,
    ) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        x_c = ((x1 + x2) / 2) / image_w
        y_c = ((y1 + y2) / 2) / image_h
        w = (x2 - x1) / image_w
        h = (y2 - y1) / image_h
        return (
            max(0.0, min(1.0, x_c)),
            max(0.0, min(1.0, y_c)),
            max(0.0, min(1.0, w)),
            max(0.0, min(1.0, h)),
        )

    # --- Single image generation ---

    def _generate_one(
        self,
        background: np.ndarray,
        compositor: SceneCompositor,
        augmentor: ImageAugmentor,
    ) -> Tuple[np.ndarray, List[Tuple[int, str, Tuple[float,float,float,float]]]]:
        """
        Generate one synthetic image.
        Returns (image_bgr, [(class_id, class_name, (xc,yc,w,h)), ...]).
        """
        cfg = self._config
        w_out, h_out = cfg.output_width, cfg.output_height
        bg_h, bg_w = background.shape[:2]
        bg = cv2.resize(background, (w_out, h_out))

        annotations: List[Tuple[int, str, Tuple[float,float,float,float]]] = []
        placed_boxes: List[Tuple[int, int, int, int]] = []
        composite = bg.copy()

        # Decide if negative sample
        is_negative = random.random() < cfg.negative_sample_ratio
        num_instances = 0 if is_negative else random.randint(cfg.instances_min, cfg.instances_max)

        for _ in range(num_instances):
            product = self._sample_product()
            renderer = self._get_renderer(product)

            # Random rotation
            rx = random.uniform(*cfg.rotation.x_range)
            ry = random.uniform(*cfg.rotation.y_range)
            rz = random.uniform(*cfg.rotation.z_range)

            try:
                rgba = renderer.render(rx, ry, rz, product.render_size)
            except Exception as exc:
                continue  # skip this instance on render failure

            # Random placement zone and position
            zone = random.choice(cfg.placement_zones)
            cx_orig, cy_orig = self._sample_position(zone)
            
            # Map coordinates to output size
            cx = int(cx_orig * w_out / bg_w)
            cy = int(cy_orig * h_out / bg_h)

            # Scale the object appropriately relative to the background resize
            bg_scale_x = w_out / bg_w
            bg_scale_y = h_out / bg_h
            bg_scale = (bg_scale_x + bg_scale_y) / 2.0
            scale = cfg.fixed_scale * bg_scale

            # Composite and get bbox
            composited, bbox = compositor.composite(
                composite, rgba, cx, cy, scale, cfg.edge_feather_px
            )
            if bbox is None:
                continue

            if not self._check_visible(bbox, rgba, scale):
                continue

            if self._check_overlap(bbox, placed_boxes):
                continue

            composite = composited
            placed_boxes.append(bbox)
            yolo_box = self._bbox_to_yolo(bbox, w_out, h_out)
            annotations.append((product.class_id, product.class_name, yolo_box))

        # Apply augmentations
        composite = augmentor.augment(composite)
        return composite, annotations

    # --- Main generation loop ---

    def generate(
        self,
        output_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> GenerationResult:
        """
        Generate the full dataset. Writes images + YOLO labels + data.yaml.
        Optionally registers a dataset in TrainingStore.
        """
        cfg = self._config
        result = GenerationResult()
        result.total = cfg.num_images
        result.output_dir = output_dir

        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Load background
        background = cv2.imread(cfg.background_path)
        if background is None:
            raise RuntimeError(f"Cannot load background image: {cfg.background_path}")

        compositor = SceneCompositor(cfg)
        augmentor = ImageAugmentor(cfg.augmentation)

        # Sorted unique classes
        classes = sorted(cfg.products, key=lambda p: p.class_id)

        try:
            for i in range(cfg.num_images):
                if self._stop_event.is_set():
                    break

                try:
                    image, annotations = self._generate_one(background, compositor, augmentor)
                except Exception as exc:
                    result.failed += 1
                    result.errors.append(f"Image {i}: {exc}")
                    continue

                # Save image
                img_name = f"{i:06d}.jpg"
                img_path = os.path.join(images_dir, img_name)
                cv2.imwrite(img_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                # Save YOLO label
                lbl_name = f"{i:06d}.txt"
                lbl_path = os.path.join(labels_dir, lbl_name)
                with open(lbl_path, "w", encoding="utf-8") as f:
                    for class_id, _, (xc, yc, w, h) in annotations:
                        f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

                result.generated += 1
                if progress_callback:
                    progress_callback(result.generated, cfg.num_images)

        finally:
            self._close_renderers()

        # Write data.yaml
        class_names = [c.class_name for c in classes]
        yaml_path = os.path.join(output_dir, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "train": images_dir,
                "val": images_dir,
                "nc": len(class_names),
                "names": class_names,
            }, f, sort_keys=False)

        # Register in TrainingStore
        if self._store is not None:
            try:
                result.dataset_id = self._register_dataset(images_dir, labels_dir, classes)
            except Exception as exc:
                result.errors.append(f"TrainingStore registration failed: {exc}")

        return result

    def _register_dataset(
        self,
        images_dir: str,
        labels_dir: str,
        classes: List[ProductClass],
    ) -> str:
        """Create a dataset + images + annotations in the existing TrainingStore."""
        cfg = self._config
        store = self._store

        dataset = store.create_dataset(cfg.dataset_name, cfg.dataset_description)
        dataset_id = dataset["id"]

        # Add classes
        colors = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
                  "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4"]
        for idx, product in enumerate(classes):
            color = colors[idx % len(colors)]
            try:
                store.add_class(dataset_id, product.class_name, color, class_id=product.class_id)
            except ValueError:
                pass  # already exists

        # Add images and annotations
        version = store.create_annotation_version(dataset_id, "v1-synthetic")
        version_id = version["id"]

        for img_filename in sorted(os.listdir(images_dir)):
            if not img_filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(img_filename)[0]
            img_path = os.path.join(images_dir, img_filename)
            lbl_path = os.path.join(labels_dir, f"{stem}.txt")

            # Add image via store
            with open(img_path, "rb") as f:
                fs = FileStorage(stream=f, filename=img_filename, content_type="image/jpeg")
                saved = store.add_images(dataset_id, [fs])

            if not saved:
                continue

            image_id = saved[0]["id"]

            # Parse and add annotations
            annotations = []
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            annotations.append({
                                "class_id": int(parts[0]),
                                "x": float(parts[1]),
                                "y": float(parts[2]),
                                "w": float(parts[3]),
                                "h": float(parts[4]),
                            })
            if annotations:
                store.save_annotations(dataset_id, version_id, image_id, annotations)

        return dataset_id

    def preview(self, count: int = 6) -> List[np.ndarray]:
        """Generate a small batch of preview images (no file output)."""
        cfg = self._config
        background = cv2.imread(cfg.background_path)
        if background is None:
            raise RuntimeError(f"Cannot load background: {cfg.background_path}")

        compositor = SceneCompositor(cfg)
        augmentor = ImageAugmentor(cfg.augmentation)
        results = []
        try:
            for _ in range(count):
                image, _ = self._generate_one(background, compositor, augmentor)
                results.append(image)
        finally:
            self._close_renderers()
        return results

    def render_calibration_image(
        self,
        product_id: int,
        x: int,
        y: int,
        rx: float,
        ry: float,
        rz: float
    ) -> np.ndarray:
        """
        Renders a single specific product at exact rotation angles and exact position.
        Uses the ScaleMap to determine scale at Y.
        Returns the BGR composited image.
        """
        cfg = self._config
        background = cv2.imread(cfg.background_path)
        if background is None:
            raise RuntimeError(f"Cannot load background: {cfg.background_path}")

        # Find product
        product = next((p for p in cfg.products if p.class_id == product_id), None)
        if product is None:
            if cfg.products:
                product = cfg.products[0]
            else:
                raise ValueError("No products configured for calibration.")

        compositor = SceneCompositor(cfg)
        renderer = self._get_renderer(product)
        
        try:
            rgba = renderer.render(rx, ry, rz, product.render_size)
        except Exception as exc:
            raise RuntimeError(f"Calibration render failed: {exc}") from exc
        finally:
            self._close_renderers()

        scale = cfg.fixed_scale
        
        bg_resized = background.copy()

        composited, bbox = compositor.composite(
            bg_resized, rgba, x, y, scale, cfg.edge_feather_px
        )

        # Draw a small crosshair and label to show where the user clicked
        cv2.drawMarker(composited, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        cv2.putText(composited, f"Scale: {scale:.2f}", (x + 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        return composited
