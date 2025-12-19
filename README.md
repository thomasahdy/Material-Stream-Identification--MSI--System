# Material Stream Identification (MSI) System

Automated waste material classification using Machine Learning (SVM and k-NN).

## 🎯 Results

| Model | Accuracy | Target |
|-------|----------|--------|
| **SVM** | **89.3%** | 85% ✅ |
| k-NN | 83.6% | 85% |

## 📁 Project Structure

```
MachinLearning/
├── MSI_System.ipynb      # Training notebook
├── app.py                # Flask web application
├── feature_utils.py      # Feature extraction module
├── requirements.txt      # Dependencies
├── models/
│   ├── svm_model.pkl     # Trained SVM model
│   └── knn_model.pkl     # Trained k-NN model
├── templates/
│   ├── index.html        # Upload page
│   └── camera.html       # Camera page
├── static/
│   └── style.css         # Styling
└── dataset/              # Training images
    ├── glass/
    ├── paper/
    ├── cardboard/
    ├── plastic/
    ├── metal/
    └── trash/
```

## 🚀 Quick Start

### 1. Setup Environment
```powershell
cd MachinLearning
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Models (if needed)
```powershell
jupyter notebook MSI_System.ipynb
# Run all cells
```

### 3. Run Web App
```powershell
python app.py
# Open http://localhost:5000
```

## 🔬 Technical Details

### Feature Extraction
- **HOG** (Histogram of Oriented Gradients) - Shape features
- **Color Histogram** (RGB, 32 bins) - Color distribution
- **LBP** (Local Binary Pattern) - Texture features

### Classifiers
- **SVM**: RBF kernel, C=10, PCA(0.95), class_weight='balanced'
- **k-NN**: distance-weighted, n_neighbors=5

### Unknown Handling
Low confidence predictions (< 60%) are classified as "Unknown".

## 📝 Material Classes

| ID | Class |
|----|-------|
| 0 | Glass |
| 1 | Paper |
| 2 | Cardboard |
| 3 | Plastic |
| 4 | Metal |
| 5 | Trash |
| 6 | Unknown |

## 🛠️ Dependencies

- numpy
- opencv-python
- scikit-learn
- scikit-image
- flask
- pillow
- matplotlib
- seaborn
- joblib

## 👥 Team

Machine Learning Project - Material Stream Identification System

## 📄 License

Educational project for waste material classification.
