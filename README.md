---
title: DogBreed Vision API
emoji: 🐕
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "1.0.0"
app_file: app.py
pinned: false
---

# 🔬 DogBreed Vision API

FastAPI backend for real-time canine breed recognition powered by YOLOv8. The project is designed for Hugging Face Spaces (Docker runtime) and serves a clean REST interface ready for portfolio demonstrations and production-grade integrations.

## 🎯 Overview

- YOLOv8n model trained on the Stanford Dogs dataset with 120 breeds.
- High-performance inference with optimized CPU settings.
- Configurable thresholds and filters via `config.yaml`.
- REST endpoints for health checks, class listing, and predictions.
- Ready-to-deploy Dockerfile tailored for Hugging Face Spaces.

### Key Metrics

- **mAP50-95**: 84.3%
- **Precision**: 80.6%
- **Recall**: 76.3%
- **Training Epochs**: 164 (early stopped)

## 🚀 Quickstart (Local)

**Requirements**

- Python 3.11+
- Git LFS enabled for downloading `weights/best.pt`

```bash
git clone https://github.com/sidnei-almeida/analise_canina_yolo.git
cd analise_canina_yolo
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

The API will be available at `http://localhost:7860`. Open `http://localhost:7860/docs` to explore the automatically generated Swagger UI.

## 🧠 API Endpoints

| Method | Route            | Description                                     |
| ------ | ---------------- | ----------------------------------------------- |
| GET    | `/`              | Welcome message + link to documentation.        |
| GET    | `/health`        | Model/device status and number of classes.      |
| GET    | `/classes`       | Mapping of YOLO class IDs to breed names.       |
| POST   | `/predict`       | Run inference on an uploaded image file.        |
| POST   | `/reload-config` | Reload thresholds and filters from `config.yaml`. |

### Example Prediction Request

```bash
curl -X POST "http://localhost:7860/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@images/German_Shepherd_sample.jpg"
```

Response snippet:

```json
{
  "detections": [
    {
      "class_id": 9,
      "class_name": "german_shepherd_dog",
      "confidence": 0.8943,
      "confidence_label": "89.4%",
      "bounding_box": {
        "x_min": 122.1,
        "y_min": 88.6,
        "x_max": 401.3,
        "y_max": 476.0
      }
    }
  ],
  "inference_time": 0.162,
  "image_size": {"width": 640, "height": 480},
  "device": "cpu",
  "model_name": "yolov8n"
}
```

## ⚙️ Configuration

Fine-tune inference behavior through `config.yaml`:

- `detection.confidence_threshold`: minimum confidence for detections.
- `detection.iou_threshold`: non-maximum suppression IoU.
- `detection.max_detections`: limit per image.
- `filters.enable_class_filter`: enable whitelist filtering.
- `filters.allowed_classes`: list of class IDs to keep.
- `filters.class_specific_confidence`: override confidence per class.
- `performance.device`: `cpu` (default) or `cuda` if GPU is available.
- `performance.use_half_precision`: set to `true` for FP16 on CUDA.
- `security.max_image_dimension`: rejects very large images to protect resources.

Reload changes at runtime by calling `POST /reload-config`.

## 📁 Project Structure

```
analise_canina_yolo/
├── app.py                  # FastAPI application
├── Dockerfile              # Hugging Face Spaces (Docker) entry point
├── requirements.txt        # Python dependencies
├── run.sh                  # Local helper script (virtualenv + uvicorn)
├── config.yaml             # Inference configuration
├── args/args.yaml          # YOLO training hyperparameters (reference)
├── weights/best.pt         # Trained YOLOv8 model weights (Git LFS)
├── results/                # Training outputs (plots, metrics, CSV)
└── images/                 # Sample test images
```

## ☁️ Deploying on Hugging Face Spaces

1. Create a new **Space** and select **Docker** as the runtime.
2. Point the Space to this repository (Git integration).
3. Ensure the Space has access to the `weights/best.pt` file (Git LFS pull).
4. No additional commands are required: the provided `Dockerfile` installs dependencies and launches `uvicorn` on port `7860`.

Optional checks before pushing:

```bash
python check_deploy.py
```

## 🧰 Tech Stack

- **Model**: YOLOv8n (Ultralytics)
- **Framework**: FastAPI + Uvicorn
- **Deep Learning**: PyTorch (CPU build)
- **Image Processing**: Pillow, OpenCV-headless
- **Configuration**: YAML-based runtime tuning

## 📊 Dataset & Training

- **Dataset**: Stanford Dogs (20k+ images, 120 breeds)
- **Input Size**: 640×640
- **Augmentations**: Mosaic, HSV shifts, flips, scaling, translation
- **Training Details**: Early stopping at epoch 164, learning rate 0.01, momentum 0.937, weight decay 0.0005

Full training logs and plots are available inside the `results/` folder.

## 🤝 Contributing

Suggestions, issues, and PRs are welcome. Potential contributions include:

- Extending the API with batch inference.
- Adding support for alternate deployment targets.
- Improving documentation or localization.
- Experimenting with larger YOLO variants.

## 📄 License

This project is distributed under the MIT License.

## 👤 Author

**Sidnei Almeida**  
GitHub: [@sidnei-almeida](https://github.com/sidnei-almeida)

---

Built for high-impact portfolios and real-world computer vision integrations. 🐕

