# 🔬 DogBreed Vision - Professional Canine Breed Recognition System

A computer vision portfolio project using YOLOv8 to detect and classify dog breeds in images with high accuracy.

## 🎯 About the Project

**DogBreed Vision** is a professional canine breed recognition system based on deep learning, trained on the Stanford Dogs Dataset. Using the YOLOv8n (nano) architecture, the model can identify **120 different breeds** of dogs with high accuracy and speed.

### 🌟 Portfolio Features

- Interactive web interface with Streamlit
- Image carousel for model testing
- Real-time analysis with visual feedback
- Metric visualizations with Plotly
- Premium dark professional design

### 📊 Performance Metrics

- **mAP50-95**: 84.3%
- **Precision**: 80.6%
- **Recall**: 76.3%
- **mAP50**: 84.5%
- **Training Epochs**: 164 (with early stopping)

## 🚀 How to Run

### Prerequisites

- Python 3.10+ (Recommended: Python 3.11 or 3.12)
- pip

> **⚠️ Note:** If you're using Python 3.13+, all packages have been updated to compatible versions.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sidnei-almeida/analise_canina_yolo.git
cd analise_canina_yolo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

4. Access in browser: `http://localhost:8501`

## 📁 Project Structure

```
analise_canina_yolo/
├── app.py                  # Main Streamlit application
├── config.yaml             # Model configuration and thresholds
├── requirements.txt        # Project dependencies
├── README.md              # Documentation
├── .gitignore            # Git ignored files
├── run.sh                # Quick execution script
├── .streamlit/
│   └── config.toml       # Custom Streamlit theme
├── args/
│   └── args.yaml         # Training arguments
├── weights/
│   ├── best.pt          # Best trained model
│   └── last.pt          # Last checkpoint
├── results/
│   ├── *.png            # Curves and confusion matrices
│   ├── *.jpg            # Training/validation batches
│   └── results.csv      # Training history
└── images/              # Test images (add your images here)
```

## 🎨 Features

### 🏠 Home Page
- Project overview
- Main metrics highlighted
- Detection examples
- Training samples

### 📊 Results Analysis
- Interactive training evolution charts
- Loss visualization (training and validation)
- Precision vs Recall analysis
- Confusion matrices
- PR, P, R, and F1 curves

### 🔮 Test Model
- Test with pre-loaded images
- Upload new images
- Real-time detection
- Results visualization with bounding boxes
- Confidence for each prediction

### ℹ️ About the Model
- Complete technical specifications
- Training hyperparameters
- Data augmentation used
- Detailed final metrics
- Use cases and applications

## 🔬 Technologies Used

- **YOLOv8n**: Optimized object detection model
- **PyTorch**: Deep learning framework
- **Streamlit**: Interactive web interface
- **Plotly**: Interactive visualizations
- **OpenCV**: Image processing
- **Stanford Dogs Dataset**: Training dataset

## 📸 How to Test the Model

1. Add PNG images of dogs to the `images/` folder
2. Access the "🔮 Test Model" section in the app
3. Select an image or upload one
4. View detections in real-time

## ⚙️ Threshold Configuration

The `config.yaml` file allows you to adjust all model parameters without modifying the code:

### Detection Parameters
```yaml
detection:
  confidence_threshold: 0.25    # Minimum confidence (0.0 - 1.0)
  iou_threshold: 0.45           # IoU for NMS
  max_detections: 10            # Maximum detections per image
  image_size: 640               # Input size (pixels)
```

### Visualization
```yaml
visualization:
  line_thickness: 2             # Bounding box line thickness
  show_labels: true             # Show labels
  show_confidence: true         # Show confidence
  confidence_format: "percentage"  # Confidence format
```

### Performance
```yaml
performance:
  use_half_precision: false     # Use FP16 (GPU)
  device: "cpu"                 # Device: cpu, cuda, cuda:0, etc.
```

### Debug
```yaml
debug:
  show_inference_time: true     # Show inference time
  save_predictions: false       # Save predictions
  verbose: false                # Verbose mode
```

**After modifying `config.yaml`:**
- Changes are automatically applied on the next prediction
- Use the "🔄 Reload Config" button in the sidebar to force update
- Check active settings in the sidebar

## 🎯 Applications

- **Veterinary**: Quick breed identification in clinics
- **Shelters**: Automatic animal cataloging
- **Mobile Apps**: Integration in pet care applications
- **Education**: Learning tool about canine breeds

## 📝 Dataset

The model was trained on the **Stanford Dogs Dataset**, which contains:
- 120 different dog breeds
- More than 20,000 images
- High diversity of poses and environments
- High-quality annotations

## 🏗️ Model Architecture

- **Base Model**: YOLOv8n (Nano)
- **Input Size**: 640x640 pixels
- **Classes**: 120 breeds
- **Framework**: Ultralytics YOLOv8

## 📈 Hyperparameters

- **Epochs**: 200 (early stopped at 164)
- **Patience**: 15 epochs
- **Learning Rate**: 0.01
- **Momentum**: 0.937
- **Weight Decay**: 0.0005
- **Batch Size**: Auto

## 🚀 Deploy

### Streamlit Cloud (Recommended)

This project is optimized for deployment on Streamlit Cloud:

1. **Fork this repository**
2. **Access:** [share.streamlit.io](https://share.streamlit.io)
3. **Configure:**
   - Repository: `your-username/analise_canina_yolo`
   - Branch: `main`
   - Main file: `app.py`
   - Python version: `3.11`
4. **Deploy!**

📖 **Complete guide:** See [DEPLOY.md](DEPLOY.md) for detailed instructions

### Deployment Requirements

- ✅ PyTorch **CPU-only** (already configured in `requirements.txt`)
- ✅ OpenCV **headless** (without GUI)
- ✅ Git LFS for large models (`weights/best.pt`)
- ✅ CPU-optimized settings

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Improve documentation
- Add new visualizations

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Sidnei Almeida**
- GitHub: [@sidnei-almeida](https://github.com/sidnei-almeida)

Developed with ❤️ for intelligent canine breed recognition

---

**🐕 Canine AI** - Powered by YOLOv8 & Streamlit

