import os
import cv2
import numpy as np
import pywt
import mediapipe as mp
import matplotlib.pyplot as plt

# [CONTROL] Stable Environment
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mp_face_mesh = mp.solutions.face_mesh

def get_aligned_face(img, face_mesh):
    try:
        height, width = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        results = face_mesh.process(img_rgb)
        
        if not results or not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        # Landmarks: 33 (Left Eye), 263 (Right Eye)
        left_eye = np.array([landmarks[33].x * width, landmarks[33].y * height])
        right_eye = np.array([landmarks[263].x * width, landmarks[263].y * height])

        # Calc Rotation Matrix
        dY, dX = (right_eye[1] - left_eye[1]), (right_eye[0] - left_eye[0])
        angle = np.degrees(np.arctan2(dY, dX))
        center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
        
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        # REPLICATE padding prevents black artifacts that ruin DWT
        aligned_img = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Crop based on landmarks to remove background noise (Optional but highly recommended)
        return cv2.resize(aligned_img, (48, 48))
    except:
        return None

def apply_pro_pipeline(image_path, face_mesh):
    """
    HYBRID DIP PIPELINE: GAMMA -> ALIGN -> CLAHE -> Z-SCORE
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    # 1. Gamma Correction (Standardizes illumination across different cameras)
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)

    # 2. Geometric Alignment
    img = get_aligned_face(img, face_mesh)
    if img is None: return None
    
    # 3. CLAHE (Local Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # 4. Normalization (Z-Score)
    img_float = img.astype('float32')
    img_std = (img_float - img_float.mean()) / (img_float.std() + 1e-7)

    return img_std

def load_dataset(base_path):
    categories = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    data, labels = [], []
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    
    print(f"\n[STEP 1] Execution Started...")
    for idx, cat in enumerate(categories):
        folder_path = os.path.join(base_path, cat)
        if not os.path.exists(folder_path): continue
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  > Category: {cat} ({len(files)} files)")
        
        for img_name in files:
            processed = apply_pro_pipeline(os.path.join(folder_path, img_name), face_mesh)
            if processed is not None:
                data.append(processed.flatten())
                labels.append(idx)

    face_mesh.close()
    return np.array(data), np.array(labels)

def extract_wavelet_features(data, wavelet='db1', level=1):
    wavelet_features = []
    for img_flat in data:
        img_2d = img_flat.reshape(48, 48)
        # Using db1 (Haar)
        coeffs = pywt.wavedec2(img_2d, wavelet, level=level)
        LL = coeffs[0] 
        wavelet_features.append(LL.flatten())
    return np.array(wavelet_features)

# --- NEW VISUALIZATION FUNCTIONS ---

def visualize_full_pipeline_grid(base_path):
    categories = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    
    stages = ['Original', 'Gamma', 'Aligned', 'Final CLAHE']
    num_cats = len(categories)
    num_stages = len(stages)

    fig, axes = plt.subplots(num_cats, num_stages, figsize=(14, 18))
    
    for i, cat in enumerate(categories):
        cat_folder = os.path.join(base_path, cat)
        if not os.path.exists(cat_folder) or len(os.listdir(cat_folder)) == 0:
            continue
            
        # Pick the first image in each folder for the grid
        img_name = [f for f in os.listdir(cat_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][0]
        img_path = os.path.join(cat_folder, img_name)
        img_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 1. Original
        axes[i, 0].imshow(img_orig, cmap='gray')
        axes[i, 0].set_ylabel(cat.upper(), fontweight='bold', fontsize=12)
        if i == 0: axes[i, 0].set_title(stages[0], fontweight='bold')
        
        # 2. Gamma
        gamma = 1.2
        invGamma = 1.0 / gamma
        table = np.array([((j / 255.0) ** invGamma) * 255 for j in np.arange(0, 256)]).astype("uint8")
        img_gamma = cv2.LUT(img_orig, table)
        axes[i, 1].imshow(img_gamma, cmap='gray')
        if i == 0: axes[i, 1].set_title(stages[1], fontweight='bold')
        
        # 3. Aligned
        img_aligned = get_aligned_face(img_gamma, face_mesh)
        if img_aligned is not None:
            axes[i, 2].imshow(img_aligned, cmap='gray')
        if i == 0: axes[i, 2].set_title(stages[2], fontweight='bold')
        
        # 4. Final Output (CLAHE)
        if img_aligned is not None:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_final = clahe.apply(img_aligned)
            axes[i, 3].imshow(img_final, cmap='gray')
        if i == 0: axes[i, 3].set_title(stages[3], fontweight='bold')

        # Formatting
        for j in range(num_stages):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.suptitle("End-to-End DIP Forensic Pipeline", fontsize=16, y=0.96)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    face_mesh.close()

def visualize_pipeline_steps(image_path):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return

    # Step 1: Gamma
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(img, table)

    # Step 2: Alignment
    img_aligned = get_aligned_face(img_gamma, face_mesh)
    if img_aligned is None: return

    # Step 3: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_final = clahe.apply(img_aligned)

    titles = ['Original Gray', 'Gamma Corrected', 'MediaPipe Aligned', 'Final CLAHE']
    images = [img, img_gamma, img_aligned, img_final]

    plt.figure(figsize=(15, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    face_mesh.close()

def visualize_all_categories(base_path):
    categories = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    plt.figure(figsize=(20, 10))
    for i, cat in enumerate(categories):
        cat_folder = os.path.join(base_path, cat)
        if os.path.exists(cat_folder) and len(os.listdir(cat_folder)) > 0:
            img_path = os.path.join(cat_folder, os.listdir(cat_folder)[0])
            processed = apply_pro_pipeline(img_path, face_mesh)
            if processed is not None:
                plt.subplot(2, 4, i+1)
                plt.imshow(processed.reshape(48, 48), cmap='gray')
                plt.title(cat.upper())
                plt.axis('off')
    plt.suptitle("Processed Samples Across All Categories", fontsize=16)
    plt.tight_layout()
    plt.show()
    face_mesh.close()