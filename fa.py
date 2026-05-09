import streamlit as st
import numpy as np
import cv2
import joblib
import os
import mediapipe as mp
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# Importing logic from your core modules
from prep import apply_pro_pipeline, extract_wavelet_features
from fa import apply_pca_transform

# Page Configuration
st.set_page_config(page_title="Pharos University | Emotion AI", layout="wide")

# Professional Dark Theme UI
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stButton>button { width: 100%; background-color: #238636; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_essentials():
    """Load pre-trained PCA and Random Forest Pipeline"""
    try:
        pca = joblib.load('pca_model_v2.pkl')
        model = joblib.load('emotion_pipeline_v2.pkl')
        return pca, model
    except Exception as e:
        st.error(f"Critical Error: Model files not found. {e}")
        return None, None

def main():
    # Application Header
    st.title("PHAROS UNIVERSITY | EMOTION RECOGNITION V2.1")
    st.markdown("### Forensic Pattern Analysis System")
    st.write("---")

    # Sidebar Architecture Info
    st.sidebar.header("PROJECT ARCHITECT")
    st.sidebar.info("ARCHITECT: HAMSA\n\nFACULTY: COMPUTER SCIENCE & AI")
    st.sidebar.markdown("---")
    st.sidebar.write("**TECHNICAL SPECIFICATIONS:**")
    st.sidebar.code("DIP: Gamma + CLAHE\nFeature: DWT (db1)\nReduction: PCA (95%)\nModel: Random Forest")

    pca_obj, emotion_model = load_essentials()
    
    if not pca_obj or not emotion_model:
        st.warning("System offline: Please upload 'pca_model_v2.pkl' and 'emotion_pipeline_v2.pkl' to the directory.")
        return

    # Layout: Input vs Results
    col_input, col_output = st.columns([1, 1.2])

    with col_input:
        st.subheader("Laboratory Input")
        uploaded_file = st.file_uploader("Upload Forensic Sample (Image)", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            # Buffer image for CV2 processing
            with open("temp_buffer.png", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            ui_img = Image.open(uploaded_file)
            st.image(ui_img, caption="Raw Input Sample", use_container_width=True)
            analyze_trigger = st.button("EXECUTE ANALYSIS PIPELINE")
        else:
            analyze_trigger = False

    if analyze_trigger:
        with st.spinner("Processing Forensic Pipeline..."):
            # Setup Face Mesh
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
            
            # [STEP 1] Apply DIP Pipeline (Gamma, Align, CLAHE, Z-Score)
            processed_img = apply_pro_pipeline("temp_buffer.png", face_mesh)
            face_mesh.close()

            if processed_img is not None:
                # [STEP 2] Feature Extraction (Wavelet)
                # Reshaping to match training format
                img_array = processed_img.flatten().reshape(1, -1)
                wav_features = extract_wavelet_features(img_array)

                # [STEP 3] PCA Projection
                pca_features = apply_pca_transform(pca_obj, wav_features)

                # [STEP 4] Model Prediction
                probs = emotion_model.predict_proba(pca_features)[0]
                categories = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
                pred_idx = np.argmax(probs)
                prediction = categories[pred_idx]
                confidence = probs[pred_idx]

                with col_output:
                    st.subheader("Analysis Output")
                    
                    # Forensic Visualization
                    st.write("**Pre-processed Forensic View:**")
                    st.image(processed_img, width=150, clamp=True)
                    
                    st.markdown(f"## RESULT: <span style='color:#2ecc71'>{prediction.upper()}</span>", unsafe_allow_html=True)
                    
                    # Confidence Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence * 100,
                        title = {'text': "Confidence Score (%)", 'font': {'size': 18}},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#58a6ff"},
                            'steps': [
                                {'range': [0, 50], 'color': '#161b22'},
                                {'range': [80, 100], 'color': '#238636'}
                            ],
                        }
                    ))
                    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # Probability Bar Chart
                    fig_bar = px.bar(x=categories, y=probs*100, title="Class Probability Map", color=probs, color_continuous_scale='Blues')
                    fig_bar.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.success("Pipeline Analysis Complete.")
            else:
                st.error("Forensic Error: No face detected. Alignment failed.")

if __name__ == "__main__":
    main()