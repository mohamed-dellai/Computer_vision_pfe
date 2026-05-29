"""
synthetic_config.py
===================
Configuration dataclasses for the synthetic dataset generation system.
Handles serialization to/from JSON, validation, and defaults.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ProductClass:
    """One product class: multiple 3D model files map to one YOLO class ID."""
    class_id: int
    class_name: str
    model_paths: List[str]          # absolute paths to .obj / .glb / .stl / .ply
    render_size: int = 512          # square render resolution before scaling
    weight: float = 1.0             # relative sampling probability

    def validate(self) -> None:
        if not self.class_name or not str(self.class_name).strip():
            raise ValueError("ProductClass.class_name is required")
        if self.class_id < 0:
            raise ValueError("ProductClass.class_id must be >= 0")
        if not self.model_paths:
            raise ValueError(f"ProductClass '{self.class_name}' has no model_paths")
        for p in self.model_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"3D model not found: {p}")
        if self.render_size < 64:
            raise ValueError("ProductClass.render_size must be >= 64")
        if self.weight <= 0:
            raise ValueError("ProductClass.weight must be > 0")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProductClass":
        return cls(
            class_id=int(d["class_id"]),
            class_name=str(d["class_name"]),
            model_paths=[str(p) for p in d.get("model_paths", [])],
            render_size=int(d.get("render_size", 512)),
            weight=float(d.get("weight", 1.0)),
        )


@dataclass
class PlacementZone:
    """Polygon (pixel coords) defining where products may be placed."""
    points: List[Tuple[int, int]]   # list of (x, y) pixel vertices

    def validate(self) -> None:
        if len(self.points) < 3:
            raise ValueError("PlacementZone must have at least 3 points")

    def to_dict(self) -> Dict[str, Any]:
        return {"points": [list(p) for p in self.points]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlacementZone":
        return cls(points=[tuple(p) for p in d["points"]])  # type: ignore[arg-type]



@dataclass
class RotationConfig:
    """Per-axis rotation ranges in degrees for 3D model rendering."""
    x_range: Tuple[float, float] = (-30.0, 30.0)    # pitch
    y_range: Tuple[float, float] = (-180.0, 180.0)  # yaw (full rotation)
    z_range: Tuple[float, float] = (-15.0, 15.0)    # roll

    def validate(self) -> None:
        for rng, name in (
            (self.x_range, "x_range"),
            (self.y_range, "y_range"),
            (self.z_range, "z_range"),
        ):
            if rng[0] > rng[1]:
                raise ValueError(f"RotationConfig.{name}: min must be <= max")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_range": list(self.x_range),
            "y_range": list(self.y_range),
            "z_range": list(self.z_range),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RotationConfig":
        return cls(
            x_range=tuple(d.get("x_range", [-30.0, 30.0])),   # type: ignore[arg-type]
            y_range=tuple(d.get("y_range", [-180.0, 180.0])), # type: ignore[arg-type]
            z_range=tuple(d.get("z_range", [-15.0, 15.0])),   # type: ignore[arg-type]
        )


@dataclass
class LightingConfig:
    """Pyrender scene lighting configuration."""
    ambient_intensity: float = 0.4
    directional_intensity: float = 3.0
    directional_angle_range: Tuple[float, float] = (-30.0, 30.0)
    intensity_variation: float = 0.3    # random ± fraction per render

    def validate(self) -> None:
        if self.ambient_intensity < 0:
            raise ValueError("LightingConfig.ambient_intensity must be >= 0")
        if self.directional_intensity < 0:
            raise ValueError("LightingConfig.directional_intensity must be >= 0")
        if not (0.0 <= self.intensity_variation <= 1.0):
            raise ValueError("LightingConfig.intensity_variation must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ambient_intensity": self.ambient_intensity,
            "directional_intensity": self.directional_intensity,
            "directional_angle_range": list(self.directional_angle_range),
            "intensity_variation": self.intensity_variation,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LightingConfig":
        return cls(
            ambient_intensity=float(d.get("ambient_intensity", 0.4)),
            directional_intensity=float(d.get("directional_intensity", 3.0)),
            directional_angle_range=tuple(d.get("directional_angle_range", [-30.0, 30.0])),  # type: ignore[arg-type]
            intensity_variation=float(d.get("intensity_variation", 0.3)),
        )


@dataclass
class AugmentationConfig:
    """Post-composite image augmentation probabilities and ranges."""
    motion_blur_prob: float = 0.3
    motion_blur_kernel_range: Tuple[int, int] = (5, 15)
    brightness_prob: float = 0.5
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    noise_prob: float = 0.3
    noise_std_range: Tuple[float, float] = (5.0, 25.0)
    hue_shift_prob: float = 0.3
    hue_shift_range: Tuple[int, int] = (-10, 10)
    jpeg_artifact_prob: float = 0.2
    jpeg_quality_range: Tuple[int, int] = (60, 90)

    # Color Harmonization
    harmonize_color: bool = True
    harmonize_strength: float = 0.5  # 0.0 to 1.0 (how much to blend towards background)

    def validate(self) -> None:
        for prob, name in (
            (self.motion_blur_prob, "motion_blur_prob"),
            (self.brightness_prob, "brightness_prob"),
            (self.noise_prob, "noise_prob"),
            (self.hue_shift_prob, "hue_shift_prob"),
            (self.jpeg_artifact_prob, "jpeg_artifact_prob"),
            (self.harmonize_strength, "harmonize_strength"),
        ):
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"AugmentationConfig.{name} must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motion_blur_prob": self.motion_blur_prob,
            "motion_blur_kernel_range": list(self.motion_blur_kernel_range),
            "brightness_prob": self.brightness_prob,
            "brightness_range": list(self.brightness_range),
            "noise_prob": self.noise_prob,
            "noise_std_range": list(self.noise_std_range),
            "hue_shift_prob": self.hue_shift_prob,
            "hue_shift_range": list(self.hue_shift_range),
            "jpeg_artifact_prob": self.jpeg_artifact_prob,
            "jpeg_quality_range": list(self.jpeg_quality_range),
            "harmonize_color": self.harmonize_color,
            "harmonize_strength": self.harmonize_strength,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AugmentationConfig":
        return cls(
            motion_blur_prob=float(d.get("motion_blur_prob", 0.3)),
            motion_blur_kernel_range=tuple(d.get("motion_blur_kernel_range", [5, 15])),       # type: ignore[arg-type]
            brightness_prob=float(d.get("brightness_prob", 0.5)),
            brightness_range=tuple(d.get("brightness_range", [0.7, 1.3])),                    # type: ignore[arg-type]
            noise_prob=float(d.get("noise_prob", 0.3)),
            noise_std_range=tuple(d.get("noise_std_range", [5.0, 25.0])),                     # type: ignore[arg-type]
            hue_shift_prob=float(d.get("hue_shift_prob", 0.3)),
            hue_shift_range=tuple(d.get("hue_shift_range", [-10, 10])),                       # type: ignore[arg-type]
            jpeg_artifact_prob=float(d.get("jpeg_artifact_prob", 0.2)),
            jpeg_quality_range=tuple(d.get("jpeg_quality_range", [60, 90])),                  # type: ignore[arg-type]
            harmonize_color=bool(d.get("harmonize_color", True)),
            harmonize_strength=float(d.get("harmonize_strength", 0.5)),
        )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    """Full configuration for one synthetic generation job."""
    background_path: str
    products: List[ProductClass]
    placement_zones: List[PlacementZone]
    fixed_scale: float = 1.0
    rotation: RotationConfig = field(default_factory=RotationConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    num_images: int = 1000
    output_width: int = 640
    output_height: int = 640
    instances_min: int = 1          # minimum products per image
    instances_max: int = 3          # maximum products per image
    negative_sample_ratio: float = 0.05
    min_visible_fraction: float = 0.70
    edge_feather_px: int = 3
    iou_overlap_limit: float = 0.30
    # TrainingStore integration
    dataset_name: str = "Synthetic Dataset"
    dataset_description: str = "Auto-generated synthetic dataset"
    # Runtime state (not serialized)
    _id: Optional[str] = field(default=None, repr=False)

    def validate(self) -> None:
        if not os.path.exists(self.background_path):
            raise FileNotFoundError(f"Background not found: {self.background_path}")
        if not self.products:
            raise ValueError("At least one ProductClass is required")
        if not self.placement_zones:
            raise ValueError("At least one PlacementZone is required")

        # Check for duplicate class IDs
        seen_ids: set = set()
        for p in self.products:
            p.validate()
            if p.class_id in seen_ids:
                raise ValueError(f"Duplicate class_id {p.class_id}")
            seen_ids.add(p.class_id)

        for z in self.placement_zones:
            z.validate()

        if not (0.01 <= self.fixed_scale <= 5.0):
            raise ValueError("fixed_scale must be between 0.01 and 5.0")
        self.rotation.validate()
        self.lighting.validate()
        self.augmentation.validate()

        if self.num_images < 1:
            raise ValueError("num_images must be >= 1")
        if self.output_width < 64 or self.output_height < 64:
            raise ValueError("output dimensions must be >= 64")
        if self.instances_min < 0:
            raise ValueError("instances_min must be >= 0")
        if self.instances_max < self.instances_min:
            raise ValueError("instances_max must be >= instances_min")
        if not (0.0 <= self.negative_sample_ratio <= 1.0):
            raise ValueError("negative_sample_ratio must be in [0, 1]")
        if not (0.1 <= self.min_visible_fraction <= 1.0):
            raise ValueError("min_visible_fraction must be in [0.1, 1.0]")
        if not (0.0 <= self.iou_overlap_limit <= 1.0):
            raise ValueError("iou_overlap_limit must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "background_path": self.background_path,
            "products": [p.to_dict() for p in self.products],
            "placement_zones": [z.to_dict() for z in self.placement_zones],
            "fixed_scale": self.fixed_scale,
            "rotation": self.rotation.to_dict(),
            "lighting": self.lighting.to_dict(),
            "augmentation": self.augmentation.to_dict(),
            "num_images": self.num_images,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "instances_min": self.instances_min,
            "instances_max": self.instances_max,
            "negative_sample_ratio": self.negative_sample_ratio,
            "min_visible_fraction": self.min_visible_fraction,
            "edge_feather_px": self.edge_feather_px,
            "iou_overlap_limit": self.iou_overlap_limit,
            "dataset_name": self.dataset_name,
            "dataset_description": self.dataset_description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SyntheticConfig":
        return cls(
            background_path=str(d["background_path"]),
            products=[ProductClass.from_dict(p) for p in d.get("products", [])],
            placement_zones=[PlacementZone.from_dict(z) for z in d.get("placement_zones", [])],
            fixed_scale=float(d.get("fixed_scale", 1.0)),
            rotation=RotationConfig.from_dict(d.get("rotation", {})),
            lighting=LightingConfig.from_dict(d.get("lighting", {})),
            augmentation=AugmentationConfig.from_dict(d.get("augmentation", {})),
            num_images=int(d.get("num_images", 1000)),
            output_width=int(d.get("output_width", 640)),
            output_height=int(d.get("output_height", 640)),
            instances_min=int(d.get("instances_min", 1)),
            instances_max=int(d.get("instances_max", 3)),
            negative_sample_ratio=float(d.get("negative_sample_ratio", 0.05)),
            min_visible_fraction=float(d.get("min_visible_fraction", 0.70)),
            edge_feather_px=int(d.get("edge_feather_px", 3)),
            iou_overlap_limit=float(d.get("iou_overlap_limit", 0.30)),
            dataset_name=str(d.get("dataset_name", "Synthetic Dataset")),
            dataset_description=str(d.get("dataset_description", "")),
        )
