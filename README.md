# Emotion Recognition System 🧠🔬

An academic project developed at **Pharos University in Alexandria (PUA)** for real-time facial expression analysis. This system implements a high-precision pipeline combining **Digital Image Processing (DIP)** with **Statistical Machine Learning**.

---

## 🚀 System Features

- **Forensic Pre-processing:** - Automated face alignment using **MediaPipe** (Ocular-based rotation) to ensure spatial consistency.
    - Contrast enhancement via **CLAHE** to reveal subtle muscle micro-expressions.
    - Non-linear luminance correction using **Gamma Correction ($\gamma=1.2$)**.
- **Hybrid Feature Engineering:** - Dimensionality reduction via **2D Discrete Wavelet Transform (DWT - Haar)** to capture structural essence.
    - Feature decorrelation and compression using **Principal Component Analysis (PCA)**, retaining 95% variance.
- **Advanced Learning Pipeline:** - Handling class imbalance with **SMOTE** (Synthetic Minority Over-sampling Technique).
    - Optimized **Random Forest Ensemble** with hyperparameter tuning via GridSearchCV.
- **High Performance:** Achieved an overall accuracy of **96.39%** and a weighted F1-score of **0.97**.

---

## 🛠️ Tech Stack

- **Core Logic:** Python 3.9+
- **Computer Vision:** OpenCV, MediaPipe
- **Mathematics & AI:** Scikit-Learn, PyWavelets, NumPy, Imbalanced-learn
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** Streamlit

---

## 📂 Project Structure

- `prep.py`: Forensic image preparation (Gamma, Alignment, CLAHE).
- `fa.py`: Feature Extraction (Wavelet decomposition & PCA transformation).
- `model.py`: The Machine Learning core (ImbPipeline, SMOTE, & Random Forest).
- `vis.py`: Diagnostic visualizations (Confusion Matrix, PCA Variance, Class Balance).
- `main.py`: The system orchestrator for training and evaluation.
- `app.py`: Real-time Streamlit dashboard for inference.

---

## ⚙️ Installation & Usage Guide

Follow these exact steps to get the project running on your local environment:

### 1. Clone the Repository
```bash
git clone [https://github.com/alaa-sawy/Emotion_Recognition-.git](https://github.com/alaa-sawy/Emotion_Recognition-.git)
cd Emotion_Recognition-
```
### 2. Install Dependencies
Ensure you have Python 3.9 or higher installed. Run the following command to install all required libraries:

```Bash
pip install -r requirements.txt
```
### 3. Training the Model (Optional)
If you have the dataset and wish to re-train the model or update the feature extraction parameters:
```Bash
python main.py
```
Note: This process will generate and save pca_model_v2.pkl and emotion_pipeline_v2.pkl to your directory.

### 4. Launch the Application (Inference)
To start the interactive web dashboard and test with your own images:
```Bash
streamlit run app.py
```
### Link Source Data :
```bash
https://www.kaggle.com/datasets/shuvoalok/ck-dataset/data?select=contempt
```
