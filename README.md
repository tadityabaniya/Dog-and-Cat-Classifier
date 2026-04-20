# 🐶 Dogs vs Cats — Image Classifier using CNN

A deep learning project that builds a **Convolutional Neural Network (CNN)** to classify images of **dogs** and **cats** using TensorFlow/Keras.

---

## 📌 Project Overview

This project demonstrates a complete machine learning pipeline for binary image classification:

| Step | Description |
|------|-------------|
| **1. Data Acquisition** | Download the Dogs vs Cats dataset from Kaggle |
| **2. Data Extraction** | Extract the ZIP archive to access training and test images |
| **3. Data Loading** | Load images using `keras.utils.image_dataset_from_directory()` |
| **4. Preprocessing** | Normalize pixel values from `[0, 255]` to `[0.0, 1.0]` |
| **5. Model Building** | Construct a Sequential CNN with Conv2D, MaxPooling2D, and Dense layers |
| **6. Compilation** | Configure with Adam optimizer and binary cross-entropy loss |
| **7. Training** | Train for 10 epochs with validation monitoring |
| **8. Testing** | Load a sample image and predict its class (Dog or Cat) |

---

## 🏗️ Model Architecture

```
┌─────────────────────────────────────────────────┐
│              Input: 256 × 256 × 3 (RGB)         │
├─────────────────────────────────────────────────┤
│  Conv2D       → 32 filters, 3×3, ReLU, same     │
│  MaxPooling2D → 2×2, stride 2  → 128×128×32     │
├─────────────────────────────────────────────────┤
│  Conv2D       → 64 filters, 3×3, ReLU, same     │
│  MaxPooling2D → 2×2, stride 2  → 64×64×64       │
├─────────────────────────────────────────────────┤
│  Conv2D       → 128 filters, 3×3, ReLU, same    │
│  MaxPooling2D → 2×2, stride 2  → 32×32×128      │
├─────────────────────────────────────────────────┤
│  Flatten      → 131,072 neurons                  │
│  Dense        → 128 neurons, ReLU                │
│  Dense        → 64 neurons, ReLU                 │
│  Dense        → 1 neuron, Sigmoid (output)       │
└─────────────────────────────────────────────────┘
```

**Total Parameters:** ~16.9 million  
**Output:** Probability between 0 and 1  
- Close to **0** → **Cat**  
- Close to **1** → **Dog**

---

## 📁 Project Structure

```
DOG/
├── Dogs_vs_Cats.ipynb              # Original Jupyter Notebook (fully documented)
├── README.md                       # This file
```

---

## 🔧 Dependencies

| Package | Purpose |
|---------|---------|
| `tensorflow` | Core deep learning framework |
| `keras` | High-level neural network API (bundled with TensorFlow) |
| `opencv-python` (`cv2`) | Image reading and preprocessing |
| `matplotlib` | Image visualization |
| `kaggle` | Dataset download from Kaggle |
| `zipfile` | Archive extraction (built-in) |

### Installation

```bash
pip install tensorflow opencv-python matplotlib kaggle
```

---

## 🚀 How to Run

### Option 1: Jupyter Notebook (Recommended for Google Colab)

1. Upload `Dogs_vs_Cats.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Configure your Kaggle API token (`kaggle.json`) for dataset download.
3. Run all cells sequentially.

### Option 2: Standalone Python Script

1. Install dependencies:
   ```bash
   pip install tensorflow opencv-python matplotlib kaggle
   ```
2. Configure Kaggle API credentials.
3. Update file paths in `Dogs_vs_Cats_Documented.py` if running locally (paths default to `/content/` for Colab).
4. Run:
   ```bash
   python Dogs_vs_Cats_Documented.py
   ```

---

## 📊 Dataset

- **Source:** [Kaggle — Dogs vs Cats](https://www.kaggle.com/datasets/moazeldsokyx/dogs-vs-cats)
- **Structure:**
  ```
  dataset/
  ├── train/
  │   ├── cats/       # Training cat images
  │   └── dogs/       # Training dog images
  └── test/
      ├── cats/       # Validation cat images
      └── dogs/       # Validation dog images
  ```
- **Image Size:** All images are resized to **256 × 256** pixels during loading.
- **Batch Size:** 32 images per batch.
- **Labels:** Automatically inferred from folder names (`cats` = 0, `dogs` = 1).

---

## 📈 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Metric | Accuracy |
| Epochs | 10 |
| Batch Size | 32 |
| Input Shape | 256 × 256 × 3 |
| Normalization | Pixel values / 255.0 |

---

## 🧪 Testing / Inference

To classify a new image:

```python
import cv2
import numpy as np

# 1. Load the image
test_img = cv2.imread('path/to/image.jpg')

# 2. Resize to match training dimensions
test_img = cv2.resize(test_img, (256, 256))

# 3. Reshape to add batch dimension: (1, 256, 256, 3)
test_input = test_img.reshape((1, 256, 256, 3))

# 4. Normalize pixel values (important!)
test_input = test_input / 255.0

# 5. Predict
prediction = model.predict(test_input)

# 6. Interpret result
if prediction[0][0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")
```

> **Note:** The test image must be normalized (divided by 255.0) before prediction to match the preprocessing applied during training.

---

## ⚠️ Important Notes

1. **Google Colab Paths:** The notebook uses `/content/` paths which are specific to Google Colab. Update these paths if running locally.
2. **Kaggle API Token:** You need a `kaggle.json` file placed in `~/.kaggle/` to download the dataset.
3. **BGR vs RGB:** OpenCV loads images in BGR format. When displaying with matplotlib, convert using:
   ```python
   plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
   ```
4. **Overfitting:** With ~98% training accuracy but ~80% validation accuracy, the model shows signs of overfitting. Consider adding:
   - Dropout layers
   - Data augmentation
   - Early stopping
   - L2 regularization

---

## 📝 License

This project is for educational purposes.
