import os
import joblib

from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Data Processing
from prep import (
    load_dataset,
    extract_wavelet_features,
    visualize_full_pipeline_grid
)

# Feature Engineering
from fa import get_pca_model, apply_pca_transform

# Model
from model import train_seven_expressions_model

# Visualization
from vis import (
    plot_confusion_matrix,
    plot_pca_variance,
    plot_pca_scatter,
    plot_class_balance,
    plot_wavelet_decomposition
)


def main():

    print("\n" + "=" * 65)
    print("        PHAROS UNIVERSITY - EMOTION RECOGNITION")
    print("        PATTERN RECOGNITION FINAL PIPELINE")
    print("=" * 65)

    # =========================
    # [1] DATASET LOADING
    # =========================

    dataset_path = r"D:\Pattren\DB"

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path not found:\n{dataset_path}")
        return

    print("\n[STEP 1] Loading and preprocessing dataset...")

    X_raw, y_raw = load_dataset(dataset_path)

    print(f"[OK] Total Samples Loaded: {len(X_raw)}")

    # =========================
    # [2] VISUAL PIPELINE REVIEW
    # =========================

    print("\n[INFO] Displaying forensic preprocessing pipeline...")
    visualize_full_pipeline_grid(dataset_path)

    # =========================
    # [3] WAVELET FEATURES
    # =========================

    print("\n[STEP 2] Extracting wavelet features...")

    X_wavelet = extract_wavelet_features(X_raw)

    print(f"[OK] Wavelet Feature Shape: {X_wavelet.shape}")

    # =========================
    # [4] TRAIN / TEST SPLIT
    # =========================

    X_train_w, X_test_w, y_train, y_test = train_test_split(
        X_wavelet,
        y_raw,
        test_size=0.2,
        random_state=42,
        stratify=y_raw
    )

    print(f"\n[INFO] Train Samples: {len(X_train_w)}")
    print(f"[INFO] Test Samples : {len(X_test_w)}")

    # =========================
    # [5] PCA PROJECTION
    # =========================

    print("\n[STEP 3] Applying PCA projection...")

    pca_obj = get_pca_model(
        X_train_w,
        variance_threshold=0.95
    )

    X_train_pca = apply_pca_transform(pca_obj, X_train_w)
    X_test_pca = apply_pca_transform(pca_obj, X_test_w)

    print(f"[OK] PCA Reduced Dimensions: {X_train_pca.shape[1]}")

    # =========================
    # [6] MODEL TRAINING
    # =========================

    print("\n[STEP 4] Training optimized model...")

    result = train_seven_expressions_model(
        X_train_pca,
        X_test_pca,
        y_train,
        y_test
    )

    best_pipeline, y_test_eval, y_pred, y_before, y_after = result

    # =========================
    # [7] EVALUATION
    # =========================

    print("\n" + "!" * 25 + " FINAL REPORT " + "!" * 25)

    acc = accuracy_score(y_test_eval, y_pred)
    f1 = f1_score(y_test_eval, y_pred, average='macro')

    print(f"\n[FINAL ACCURACY] {acc * 100:.2f}%")
    print(f"[FINAL F1-MACRO] {f1:.4f}")

    target_names = [
        'Anger',
        'Contempt',
        'Disgust',
        'Fear',
        'Happy',
        'Sadness',
        'Surprise'
    ]

    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(
        y_test_eval,
        y_pred,
        target_names=target_names
    ))

    print("!" * 64)

    # =========================
    # [8] SAVE MODELS
    # =========================

    print("\n[STEP 5] Exporting trained models...")

    joblib.dump(pca_obj, 'pca_model_v2.pkl')
    joblib.dump(best_pipeline, 'emotion_pipeline_v2.pkl')

    print("[OK] PCA model saved.")
    print("[OK] Emotion model saved.")

    # =========================
    # [9] VISUAL ANALYTICS
    # =========================

    print("\n[VISUALS] Opening diagnostic visualizations...")

    plot_class_balance(y_before, y_after)

    plot_confusion_matrix(
        y_test_eval,
        y_pred
    )

    plot_pca_variance(pca_obj)

    if X_test_pca.shape[1] >= 2:
        plot_pca_scatter(
            X_test_pca,
            y_test_eval
        )
    else:
        print("[WARNING] PCA scatter skipped (less than 2 components).")

    # =========================
    # [10] WAVELET ANALYSIS
    # =========================

    print("\n[INFO] Displaying wavelet decomposition analysis...")

    sample_img = X_raw[0].reshape(48, 48)

    plot_wavelet_decomposition(sample_img)

    print("\n[COMPLETE] Full pipeline execution finished successfully.")


if __name__ == "__main__":
    main()

