import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import pywt

sns.set_theme(style="whitegrid")

def plot_confusion_matrix(y_true, y_pred):
    categories = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
    cm = confusion_matrix(y_true, y_pred)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako', 
                xticklabels=categories, yticklabels=categories, cbar=True)
    
    plt.title('Forensic Confusion Matrix - Emotion Recognition V1.0', fontsize=16, pad=20)
    plt.ylabel('Actual Category (Ground Truth)', fontsize=13)
    plt.xlabel('Model Prediction', fontsize=13)
    plt.show()

def plot_pca_variance(pca_model):
    variance = np.cumsum(pca_model.explained_variance_ratio_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(variance) + 1), variance, marker='o', linestyle='--', color='darkblue')
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% Threshold')
    
    plt.title('Cumulative Explained Variance (PCA Selection)', fontsize=15)
    plt.xlabel('Number of Principal Components', fontsize=12)
    plt.ylabel('Cumulative Variance', fontsize=12)
    plt.legend()
    plt.show()

def plot_pca_scatter(X_pca, y):
    categories = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, edgecolors='w', s=60)
    
    plt.legend(handles=scatter.legend_elements()[0], labels=categories, 
                title="Emotions", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title('PCA Feature Distribution (PC1 vs PC2)', fontsize=15)
    plt.xlabel(f'Principal Component 1', fontsize=12)
    plt.ylabel(f'Principal Component 2', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_class_balance(y_before, y_after):
    categories = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
    
    # 1. Convert numeric labels to emotion names for better clarity
    y_before_names = [categories[i] for i in y_before]
    y_after_names = [categories[i] for i in y_after]

    # 2. Prepare DataFrames for Seaborn
    df_after = pd.DataFrame({'Emotion': y_after_names, 'Data Type': 'Balanced (Synthetic + Original)'})
    df_before = pd.DataFrame({'Emotion': y_before_names, 'Data Type': 'Original (Imbalanced)'})

    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    # 3. Draw the full balanced distribution first (The Target)
    sns.countplot(data=df_after, x='Emotion', order=categories, 
                  color='#2ecc71', label='Synthetic Samples (SMOTE)')

    # 4. Draw the original distribution on top (The Source)
    sns.countplot(data=df_before, x='Emotion', order=categories, 
                  color='#3498db', label='Original Samples')

    # 5. Professional Formatting
    plt.title('PHAROS UNIVERSITY | SMOTE IMPACT ANALYSIS', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Emotion Category', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(fontsize=12, loc='upper right')

    # Add count labels on top of bars
    for i, p in enumerate(plt.gca().patches[:7]): 
        height = p.get_height()
        plt.gca().text(p.get_x() + p.get_width()/2., height + 5,
                f'Orig: {int(height)}', ha="center", fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_wavelet_decomposition(image_2d, wavelet='db1'):
    coeffs = pywt.wavedec2(image_2d, wavelet, level=1)
    LL, (LH, HL, HH) = coeffs

    titles = ['Approximation (LL)', 'Horizontal (LH)', 'Vertical (HL)', 'Diagonal (HH)']
    images = [LL, LH, HL, HH]

    plt.figure(figsize=(14, 4))
    for i, img in enumerate(images):
        plt.subplot(1, 4, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(titles[i], fontsize=11, fontweight='bold')
        plt.axis('off')
    plt.suptitle(f"Discrete Wavelet Transform Analysis (Wavelet: {wavelet})", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()