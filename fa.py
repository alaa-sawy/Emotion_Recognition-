import pywt
import numpy as np
from sklearn.decomposition import PCA

def extract_hybrid_wavelet_features(data, wavelet='db1', level=1):
    hybrid_features = []
    
    for img_2d in data:
        # 1. Discrete Wavelet Transform (DWT)
        coeffs = pywt.wavedec2(img_2d, wavelet, level=level)
        LL, (LH, HL, HH) = coeffs
        f_vector = np.hstack([
            LL.flatten(), 
            LH.flatten(), 
            HL.flatten()
        ])
        
        hybrid_features.append(f_vector)
        
    return np.array(hybrid_features)

def get_pca_model(feature_matrix, variance_threshold=0.98):
    # Using whiten=True to decorrelate features for better classifier performance
    pca = PCA(
        n_components=variance_threshold, 
        svd_solver='auto', 
        whiten=True, 
        random_state=42
    )
    pca.fit(feature_matrix)
    
    return pca

def apply_pca_transform(pca_model, feature_matrix):
    """Applies the learned PCA projection to new data."""
    return pca_model.transform(feature_matrix)
