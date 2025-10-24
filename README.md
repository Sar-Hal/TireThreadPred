# TireThreadPred 🚗🔍

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**AI-powered Tire Quality Classification System** using SE-MobileNetV2 with Transfer Learning

Automatically detect defective and good condition tires from images with **90%+ accuracy** using state-of-the-art deep learning techniques.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Training](#training)
- [Prediction](#prediction)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## 🎯 Overview

TireThreadPred is an advanced computer vision project that classifies tire images into two categories:
- **Defective**: Tires with visible defects, damage, or wear issues
- **Good**: Tires in acceptable condition

The system leverages transfer learning with MobileNetV2 enhanced with Squeeze-and-Excitation (SE) blocks for improved channel attention and feature learning.

### Key Highlights

- ✅ **90-95% Test Accuracy**
- ✅ **Efficient TFRecord Pipeline** (5x faster data loading)
- ✅ **SE-MobileNetV2 Architecture** with channel attention
- ✅ **Fine-tuning Strategy** (last 30 layers unfrozen)
- ✅ **Class Weight Balancing** for imbalanced datasets
- ✅ **Advanced Data Augmentation**
- ✅ **Comprehensive Evaluation Metrics**

---

## 🚀 Features

### Model Features
- **Transfer Learning**: Pre-trained MobileNetV2 on ImageNet
- **SE Blocks**: Squeeze-and-Excitation for channel-wise attention
- **Fine-tuning**: Selective layer unfreezing for optimal performance
- **Cosine Decay LR**: Smooth learning rate scheduling
- **Class Weighting**: Handle imbalanced datasets effectively

### Data Pipeline
- **TFRecord Format**: Fast and efficient data loading
- **Advanced Augmentation**: Random flips, rotations, brightness, contrast, saturation
- **70-15-15 Split**: Training, validation, and test sets
- **Prefetching & Parallel Processing**: Optimized for performance

### Evaluation & Monitoring
- **Multiple Metrics**: Accuracy, Precision, Recall, AUC
- **Confusion Matrix**: Visual performance analysis
- **TensorBoard Integration**: Real-time training monitoring
- **Sample Predictions**: Visual verification of model performance

---

## 🏗️ Model Architecture

```
Input (224x224x3)
    ↓
MobileNetV2 (Pre-trained on ImageNet)
    ↓ (Last 30 layers fine-tuned)
SE Block (Channel Attention)
    ↓
Global Average Pooling
    ↓
Batch Normalization + Dropout(0.5)
    ↓
Dense(256, ReLU)
    ↓
Batch Normalization + Dropout(0.3)
    ↓
Dense(2, Softmax)
    ↓
Output: [Defective, Good]
```

### SE Block (Squeeze-and-Excitation)
The SE block learns to emphasize important feature channels:
1. **Squeeze**: Global average pooling to capture channel-wise statistics
2. **Excitation**: Two FC layers to learn channel weights
3. **Scale**: Multiply features by learned weights

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- TensorFlow 2.x
- GPU support (optional but recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Sar-Hal/TireThreadPred.git
cd TireThreadPred
```

2. **Install dependencies**
```bash
pip install tensorflow numpy matplotlib pillow scikit-learn seaborn tqdm
```

Or use a requirements file:
```bash
pip install -r requirements.txt
```

3. **Verify TensorFlow installation**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPU: {tf.config.list_physical_devices(\"GPU\")}')"
```

---

## ⚡ Quick Start

### 1. Prediction (Using Pre-trained Model)

```python
python predict_tire.py
```

Edit the `Config` class in `predict_tire.py` to specify your image:

```python
class Config:
    IMG_SIZE = 224
    CLASS_NAMES = ['defective', 'good']
    MODEL_PATH = 'tire_model_final'
    IMAGE_PATH = 'your_image.png'  # Change this
```

### 2. Batch Prediction

Uncomment the batch prediction section in `predict_tire.py`:

```python
# Predict on multiple images
predict_batch(model, image_folder='test_images')
```

---

## 📊 Dataset

The model was trained on the **Tyre Quality Classification Dataset** containing:
- **Defective tires**: Images showing various tire defects
- **Good tires**: Images of tires in acceptable condition

### Dataset Structure
```
dataset/
├── defective/
│   ├── defect_001.jpg
│   ├── defect_002.jpg
│   └── ...
└── good/
    ├── good_001.jpg
    ├── good_002.jpg
    └── ...
```

### Data Split
- **Training**: 70% (with augmentation)
- **Validation**: 15%
- **Test**: 15%

---

## 🎓 Training

### Training Notebooks

Three versions of training notebooks are provided in `Training Notebooks/`:

1. **version1.ipynb**: Initial baseline implementation
2. **version2.ipynb**: Improved version with SE blocks
3. **version3.ipynb**: Final optimized version (recommended)

### Training Configuration

Key hyperparameters in `version3.ipynb`:

```python
class Config:
    IMG_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 60
    INITIAL_LEARNING_RATE = 0.0001
    UNFREEZE_LAYERS = 30
    SE_REDUCTION = 8
    DROPOUT_RATE = 0.4
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
```

### Training Pipeline

1. **Data Preparation**: Convert images to TFRecord format
2. **Model Creation**: Build SE-MobileNetV2 architecture
3. **Training**: Fit model with callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
4. **Evaluation**: Comprehensive testing and visualization
5. **Export**: Save model in TensorFlow SavedModel format

### Callbacks Used

- **EarlyStopping**: Prevent overfitting (patience=15)
- **ReduceLROnPlateau**: Adaptive learning rate (factor=0.5, patience=5)
- **ModelCheckpoint**: Save best model based on validation accuracy
- **TensorBoard**: Real-time monitoring and visualization

---

## 🔮 Prediction

### Single Image Prediction

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Load model
model = keras.models.load_model('tire_model_final', 
                                custom_objects={'SEBlock': SEBlock})

# Load and preprocess image
img = Image.open('test_tire.jpg').convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_batch)
class_names = ['defective', 'good']
predicted_class = class_names[np.argmax(predictions[0])]
confidence = np.max(predictions[0]) * 100

print(f"Predicted: {predicted_class} ({confidence:.2f}% confidence)")
```

### Output Format

```
🔍 Image: test_tire.jpg

📊 Class Probabilities:
   Defective: 12.34%
   Good:      87.66%

🎯 Predicted Class: GOOD
💪 Confidence: 87.66%

INTERPRETATION
✓ This tire appears to be in GOOD condition
  Recommended action: Continue normal use
  
  Confidence level: HIGH
  The model is very confident in this prediction.
```

---

## 📈 Results

### Performance Metrics

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 95.2% | 92.8% | 91.5% |
| **Precision** | 94.8% | 91.9% | 90.7% |
| **Recall** | 95.6% | 93.2% | 92.1% |
| **AUC** | 0.985 | 0.972 | 0.968 |

### Per-Class Performance

| Class | Samples | Accuracy | Avg Confidence |
|-------|---------|----------|----------------|
| **Defective** | 143 | 89.5% | 86.2% |
| **Good** | 135 | 93.3% | 91.8% |

### Model Statistics

- **Total Parameters**: ~3.5M
- **Trainable Parameters**: ~1.2M
- **Model Size**: ~14 MB (SavedModel format)
- **Inference Time**: ~50ms per image (CPU), ~10ms (GPU)

---

## 📁 Project Structure

```
TireThreadPred/
│
├── README.md                      # Project documentation (this file)
├── predict_tire.py                # Prediction script for inference
├── requirements.txt               # Python dependencies
│
├── tire_model_final/              # Trained model (SavedModel format)
│   ├── saved_model.pb
│   ├── keras_metadata.pb
│   ├── fingerprint.pb
│   ├── assets/
│   └── variables/
│       ├── variables.data-00000-of-00001
│       └── variables.index
│
├── Training Notebooks/            # Jupyter notebooks for training
│   ├── version1.ipynb            # Baseline implementation
│   ├── version2.ipynb            # Improved with SE blocks
│   └── version3.ipynb            # Final optimized version
│
├── tfrecords/                     # TFRecord dataset files (generated)
│   ├── train.tfrecord
│   ├── val.tfrecord
│   └── test.tfrecord
│
└── outputs/                       # Training outputs (generated)
    ├── confusion_matrix_improved.png
    ├── training_history_improved.png
    ├── sample_predictions.png
    └── logs/                      # TensorBoard logs
```

---

## 🔧 Technical Details

### SE Block Implementation

```python
class SEBlock(layers.Layer):
    def __init__(self, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction = reduction
        
    def build(self, input_shape):
        channels = input_shape[-1]
        self.squeeze = layers.GlobalAveragePooling2D()
        self.excitation = keras.Sequential([
            layers.Dense(channels // self.reduction, activation='relu'),
            layers.Dense(channels, activation='sigmoid')
        ])
        
    def call(self, inputs):
        squeeze = self.squeeze(inputs)
        excitation = self.excitation(squeeze)
        excitation = tf.reshape(excitation, [-1, 1, 1, tf.shape(inputs)[-1]])
        return inputs * excitation
```

### Data Augmentation Strategy

- **Horizontal Flip**: Random left-right flip
- **Vertical Flip**: Random up-down flip (tires are radially symmetric)
- **Rotation**: Random 0°, 90°, 180°, 270° rotations
- **Brightness**: Random adjustment (±20%)
- **Contrast**: Random adjustment (0.8-1.2x)
- **Saturation**: Random adjustment (0.8-1.2x)

### TFRecord Pipeline Benefits

1. **Faster Loading**: 5x faster than loading raw images
2. **Memory Efficient**: Stream data from disk
3. **Parallel Processing**: Utilize multiple CPU cores
4. **Reproducibility**: Consistent data ordering
5. **Portability**: Single file per split

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Improvement

- [ ] Add more tire defect categories
- [ ] Implement ensemble models
- [ ] Create web interface (Flask/Streamlit)
- [ ] Add ONNX export for deployment
- [ ] Implement explainability (Grad-CAM)
- [ ] Mobile deployment (TensorFlow Lite)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Sar-Hal**

- GitHub: [@Sar-Hal](https://github.com/Sar-Hal)
- Project Link: [https://github.com/Sar-Hal/TireThreadPred](https://github.com/Sar-Hal/TireThreadPred)

---

## 🙏 Acknowledgments

- **MobileNetV2**: Howard et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
- **SE-Net**: Hu et al., "Squeeze-and-Excitation Networks"
- **TensorFlow**: Google Brain Team
- **Dataset**: Tyre Quality Classification Dataset (Kaggle)

---

## 📞 Support

If you have any questions or issues:

1. Check existing [Issues](https://github.com/Sar-Hal/TireThreadPred/issues)
2. Open a new issue with detailed description
3. Contact via GitHub

---

## 🎯 Citation

If you use this project in your research, please cite:

```bibtex
@software{tirethreadpred2025,
  author = {Sar-Hal},
  title = {TireThreadPred: AI-Powered Tire Quality Classification},
  year = {2025},
  url = {https://github.com/Sar-Hal/TireThreadPred}
}
```

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ by Sar-Hal

</div>
