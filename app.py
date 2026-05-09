import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import mediapipe as mp

# Core Modules
from prep import get_aligned_face
from fa import extract_wavelet_features, apply_pca_transform

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Pharos University | Emotion AI",
    layout="wide"
)

# =========================================================
# CUSTOM DARK THEME
# =========================================================

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }

    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }

    [data-testid="stSidebar"] {
        background-color: #161b22;
    }

    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD TRAINED MODELS
# =========================================================

def load_models():

    try:
        pca = joblib.load('pca_model_v2.pkl')
        model = joblib.load('emotion_pipeline_v2.pkl')

        return pca, model

    except Exception as e:

        st.error(f"Error loading models: {e}")
        return None, None

# =========================================================
# FULL PREPROCESSING PIPELINE
# SAME AS TRAINING PIPELINE
# =========================================================

def process_uploaded_image(gray_image, face_mesh):

    # =========================
    # [1] GAMMA CORRECTION
    # =========================

    gamma = 1.2
    inv_gamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")

    gamma_corrected = cv2.LUT(
        gray_image,
        table
    )

    # =========================
    # [2] FACE ALIGNMENT
    # =========================

    aligned_face = get_aligned_face(
        gamma_corrected,
        face_mesh
    )

    if aligned_face is None:
        return None

    # =========================
    # [3] CLAHE ENHANCEMENT
    # =========================

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_face = clahe.apply(
        aligned_face
    )

    # =========================
    # [4] Z-SCORE NORMALIZATION
    # =========================

    face_float = enhanced_face.astype('float32')

    normalized_face = (
        (face_float - face_float.mean()) /
        (face_float.std() + 1e-7)
    )

    return normalized_face

# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    # =====================================================
    # HEADER
    # =====================================================

    st.title("PHAROS UNIVERSITY | EMOTION RECOGNITION")
    st.markdown("### Pattern Recognition Final Project")

    st.write("---")

    # =====================================================
    # SIDEBAR
    # =====================================================

    st.sidebar.header("PROJECT ARCHITECT")

    st.sidebar.info(
        "ARCHITECT: HAMSA\n\n"
        "FACULTY: COMPUTER SCIENCE & AI"
    )

    st.sidebar.markdown("---")

    st.sidebar.write("**TECHNICAL STACK:**")

    st.sidebar.code(
        "DWT: Daubechies (db1)\n"
        "PCA: 95% Variance\n"
        "Classifier: Random Forest + SMOTE"
    )

    # =====================================================
    # LOAD MODELS
    # =====================================================

    pca_model, emotion_model = load_models()

    if pca_model is None:

        st.warning(
            "Please ensure the trained .pkl files "
            "exist in the same directory."
        )

        return

    # =====================================================
    # LAYOUT
    # =====================================================

    col_input, col_output = st.columns([1, 1.2])

    # =====================================================
    # INPUT COLUMN
    # =====================================================

    with col_input:

        st.subheader("Input Area")

        uploaded_file = st.file_uploader(
            "Upload Image (JPG / PNG)",
            type=['jpg', 'jpeg', 'png']
        )

        if uploaded_file:

            img = Image.open(uploaded_file)

            st.image(
                img,
                caption="Original Input",
                use_container_width=True
            )

            run_analysis = st.button(
                "ANALYZE EXPRESSION",
                type="primary",
                use_container_width=True
            )

        else:

            run_analysis = False

    # =====================================================
    # ANALYSIS PIPELINE
    # =====================================================

    if run_analysis and uploaded_file:

        with st.spinner("Processing forensic pipeline..."):

            # Convert Image
            frame = np.array(
                img.convert('RGB')
            )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2GRAY
            )

            # =============================================
            # MEDIAPIPE FACEMESH
            # =============================================

            mp_face_mesh = mp.solutions.face_mesh

            with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1
            ) as face_mesh:

                # =========================================
                # FULL PREPROCESSING
                # =========================================

                processed_face = process_uploaded_image(
                    gray,
                    face_mesh
                )

                if processed_face is not None:

                    # =====================================
                    # FEATURE EXTRACTION
                    # =====================================

                    wav_features = extract_wavelet_features(
                        [processed_face.flatten()]
                    )

                    pca_features = apply_pca_transform(
                        pca_model,
                        wav_features
                    )

                    # =====================================
                    # MODEL INFERENCE
                    # =====================================

                    probs = emotion_model.predict_proba(
                        pca_features
                    )[0]

                    categories = [
                        'Anger',
                        'Contempt',
                        'Disgust',
                        'Fear',
                        'Happy',
                        'Sadness',
                        'Surprise'
                    ]

                    pred_idx = np.argmax(probs)

                    prediction = categories[pred_idx]

                    confidence = probs[pred_idx]

                    # =====================================
                    # OUTPUT COLUMN
                    # =====================================

                    with col_output:

                        st.subheader("Analysis Output")

                        st.write(
                            "**Forensic Preview "
                            "(48x48 Processed Face):**"
                        )

                        st.image(
                            processed_face,
                            width=180,
                            clamp=True
                        )

                        st.markdown(
                            f"""
                            ## OPTIMAL PREDICTION:
                            <span style='color:#2ecc71'>
                            {prediction.upper()}
                            </span>
                            """,
                            unsafe_allow_html=True
                        )

                        # =================================
                        # CONFIDENCE GAUGE
                        # =================================

                        fig_gauge = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=confidence * 100,
                                domain={
                                    'x': [0, 1],
                                    'y': [0, 1]
                                },
                                title={
                                    'text': "Confidence Score (%)",
                                    'font': {'size': 20}
                                },
                                gauge={
                                    'axis': {
                                        'range': [0, 100],
                                        'tickwidth': 1
                                    },

                                    'bar': {
                                        'color': "#58a6ff"
                                    },

                                    'steps': [
                                        {
                                            'range': [0, 50],
                                            'color': '#161b22'
                                        },

                                        {
                                            'range': [50, 80],
                                            'color': '#1f6feb'
                                        },

                                        {
                                            'range': [80, 100],
                                            'color': '#238636'
                                        }
                                    ]
                                }
                            )
                        )

                        fig_gauge.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            font={
                                'color': "white",
                                'family': "Arial"
                            },
                            height=350
                        )

                        st.plotly_chart(
                            fig_gauge,
                            use_container_width=True
                        )

                    # =====================================
                    # PROBABILITY DISTRIBUTION
                    # =====================================

                    st.markdown("---")

                    st.subheader(
                        "Forensic Probability Distribution"
                    )

                    fig_bar = px.bar(
                        x=categories,
                        y=probs * 100,

                        labels={
                            'x': 'Emotion Category',
                            'y': 'Probability (%)'
                        },

                        title=(
                            "Probability Distribution "
                            "Across All Classes"
                        ),

                        color=probs,
                        color_continuous_scale='Blues'
                    )

                    fig_bar.update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)'
                    )

                    st.plotly_chart(
                        fig_bar,
                        use_container_width=True
                    )

                    # =====================================
                    # FINAL STATUS
                    # =====================================

                    if confidence >= 0.70:

                        st.success(
                            f"""
                            Analysis Complete Successfully.

                            Prediction: {prediction}

                            Confidence:
                            {confidence * 100:.2f}%
                            """
                        )

                    elif confidence >= 0.50:

                        st.warning(
                            f"""
                            Moderate Confidence Prediction.

                            Prediction: {prediction}

                            Confidence:
                            {confidence * 100:.2f}%
                            """
                        )

                    else:

                        st.error(
                            f"""
                            Low Confidence Prediction.

                            Prediction: {prediction}

                            Confidence:
                            {confidence * 100:.2f}%

                            Recommendation:
                            Please upload a clearer face image.
                            """
                        )

                else:

                    st.error(
                        "Face alignment failed. "
                        "Please upload a clearer image."
                    )

# =========================================================
# APPLICATION ENTRY
# =========================================================

if __name__ == "__main__":
    main()

