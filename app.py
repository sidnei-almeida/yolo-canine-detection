from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).parent.resolve()
MODEL_PATH = PROJECT_ROOT / "weights" / "best.pt"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class DetectionBoundingBox(BaseModel):
    x_min: float = Field(..., description="Left coordinate of the bounding box (pixels).")
    y_min: float = Field(..., description="Top coordinate of the bounding box (pixels).")
    x_max: float = Field(..., description="Right coordinate of the bounding box (pixels).")
    y_max: float = Field(..., description="Bottom coordinate of the bounding box (pixels).")


class Detection(BaseModel):
    class_id: int = Field(..., description="Numeric class identifier returned by YOLO.")
    class_name: str = Field(..., description="Human-readable dog breed name.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1).")
    confidence_label: str = Field(..., description="Formatted confidence string matching config.")
    bounding_box: DetectionBoundingBox


class PredictionResponse(BaseModel):
    detections: List[Detection]
    inference_time: float = Field(..., description="Inference time in seconds.")
    image_size: Dict[str, int] = Field(..., description="Width and height of the processed image.")
    device: str = Field(..., description="Device used by the YOLO model.")
    model_name: str = Field(..., description="Name of the underlying YOLO model.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: int
    device: str


class ModelService:
    """Service responsible for loading the YOLO model and running predictions."""

    def __init__(self) -> None:
        self._config = self._load_config()
        self._model = self._load_model()

    @property
    def config(self) -> Dict:
        return self._config

    @property
    def model(self) -> YOLO:
        return self._model

    def reload_config(self) -> Dict:
        """Reload configuration from disk."""
        self._config = self._load_config()
        return self._config

    def _load_config(self) -> Dict:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as file:
                return yaml.safe_load(file) or {}

        return {
            "detection": {
                "confidence_threshold": 0.35,
                "iou_threshold": 0.50,
                "max_detections": 10,
                "image_size": 640,
            },
            "filters": {
                "enable_class_filter": False,
                "allowed_classes": [],
                "class_specific_confidence": {},
            },
            "performance": {
                "use_half_precision": False,
                "device": "cpu",
            },
            "security": {
                "check_image_dimensions": True,
                "max_image_dimension": 4096,
                "min_image_dimension": 32,
            },
            "visualization": {
                "confidence_format": "percentage",
            },
        }

    def _load_model(self) -> YOLO:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Upload weights before running the API.")

        model = YOLO(str(MODEL_PATH))
        device = self._config.get("performance", {}).get("device", "cpu")
        try:
            model.to(device)
        except Exception as exc:
            raise RuntimeError(f"Failed to move YOLO model to device '{device}': {exc}") from exc
        return model

    def predict(self, image: Image.Image) -> PredictionResponse:
        config = self._config
        detection_cfg = config.get("detection", {})
        filters_cfg = config.get("filters", {})
        security_cfg = config.get("security", {})
        visualization_cfg = config.get("visualization", {})

        if security_cfg.get("check_image_dimensions", True):
            max_dim = security_cfg.get("max_image_dimension", 4096)
            min_dim = security_cfg.get("min_image_dimension", 32)
            width, height = image.size
            if max(width, height) > max_dim:
                raise HTTPException(
                    status_code=422,
                    detail=f"Image dimensions {width}x{height} exceed the maximum allowed size of {max_dim}px.",
                )
            if min(width, height) < min_dim:
                raise HTTPException(
                    status_code=422,
                    detail=f"Image dimensions {width}x{height} are smaller than the minimum allowed size of {min_dim}px.",
                )

        np_image = np.array(image.convert("RGB"))

        class_filter_enabled = filters_cfg.get("enable_class_filter", False)
        allowed_classes = set(filters_cfg.get("allowed_classes") or [])
        class_specific_conf = filters_cfg.get("class_specific_confidence") or {}

        threshold = float(detection_cfg.get("confidence_threshold", 0.35))
        iou = float(detection_cfg.get("iou_threshold", 0.50))
        max_det = int(detection_cfg.get("max_detections", 10))
        img_size = int(detection_cfg.get("image_size", 640))
        use_half = bool(config.get("performance", {}).get("use_half_precision", False))

        model_device = self._model.device
        half_precision = use_half and getattr(model_device, "type", "cpu") != "cpu"

        start_time = time.perf_counter()
        results = self._model.predict(
            source=np_image,
            conf=threshold,
            iou=iou,
            imgsz=img_size,
            max_det=max_det,
            half=half_precision,
            verbose=False,
        )
        inference_time = time.perf_counter() - start_time

        if not results:
            return PredictionResponse(
                detections=[],
                inference_time=inference_time,
                image_size={"width": int(image.width), "height": int(image.height)},
                device=str(model_device),
                model_name=self._model.name or "yolov8",
            )

        detections: List[Detection] = []
        names = self._model.names

        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())

            class_specific_threshold = self._resolve_class_threshold(class_specific_conf, cls_id, threshold)
            if confidence < class_specific_threshold:
                continue

            if class_filter_enabled and allowed_classes and cls_id not in allowed_classes:
                continue

            bbox = box.xyxy.tolist()[0]
            detection = Detection(
                class_id=cls_id,
                class_name=str(names.get(cls_id, f"class_{cls_id}")),
                confidence=confidence,
                confidence_label=self._format_confidence(confidence, visualization_cfg.get("confidence_format", "percentage")),
                bounding_box=DetectionBoundingBox(
                    x_min=float(bbox[0]),
                    y_min=float(bbox[1]),
                    x_max=float(bbox[2]),
                    y_max=float(bbox[3]),
                ),
            )
            detections.append(detection)

        model_config = getattr(self._model.model, "yaml", {})
        model_name = (
            (model_config.get("name") if isinstance(model_config, dict) else None)
            or getattr(self._model, "task", None)
            or "yolov8"
        )

        return PredictionResponse(
            detections=detections,
            inference_time=inference_time,
            image_size={"width": int(image.width), "height": int(image.height)},
            device=str(model_device),
            model_name=str(model_name),
        )

    @staticmethod
    def _resolve_class_threshold(class_specific_conf: Dict, class_id: int, default_threshold: float) -> float:
        if not class_specific_conf:
            return default_threshold

        # YAML may load keys as str or int; support both.
        return float(
            class_specific_conf.get(class_id)
            or class_specific_conf.get(str(class_id))
            or default_threshold
        )

    @staticmethod
    def _format_confidence(confidence: float, fmt: str) -> str:
        fmt_normalized = (fmt or "percentage").lower()
        if fmt_normalized == "decimal":
            return f"{confidence:.4f}"
        return f"{confidence * 100:.1f}%"


service = ModelService()

app = FastAPI(
    title="DogBreed Vision API",
    description=(
        "Deep learning inference API for canine breed detection using YOLOv8. "
        "Upload an image to retrieve detected dog breeds and associated metadata."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
def root() -> Dict[str, str]:
    return {
        "message": "Welcome to the DogBreed Vision API. Use POST /predict with an image file to run inference.",
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse, summary="API health status.")
def health_check() -> HealthResponse:
    model_loaded = service.model is not None
    names = service.model.names if model_loaded else {}
    return HealthResponse(
        status="ok" if model_loaded else "model-not-loaded",
        model_loaded=model_loaded,
        classes=len(names),
        device=str(service.model.device) if model_loaded else "unknown",
    )


@app.get("/classes", response_model=Dict[int, str], summary="List all available breed classes.")
def list_classes() -> Dict[int, str]:
    return {int(k): str(v) for k, v in service.model.names.items()}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Run canine breed detection on an uploaded image.",
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read the uploaded image: {exc}") from exc

    return service.predict(image)


@app.post("/reload-config", response_model=Dict, summary="Reload configuration from config.yaml.")
def reload_configuration() -> Dict:
    config = service.reload_config()
    return {"message": "Configuration reloaded successfully.", "config": config}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
