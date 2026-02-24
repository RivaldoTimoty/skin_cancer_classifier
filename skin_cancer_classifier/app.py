import os
import time

import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px

import config
from inference import load_inference_model, load_class_mapping, run_inference

# ==============================================================================
# Page Layout & Styling
# ==============================================================================
st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        color: #2C3E50;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0rem;
        padding-top: 1rem;
    }
    .sub-header {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        color: #7F8C8D;
        font-weight: 400;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .disclaimer-box {
        background-color: #FEF9E7;
        border-left: 5px solid #F1C40F;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .disclaimer-text {
        color: #D35400;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .prediction-box {
        background-color: #EAF2F8;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #D6EAF8;
        margin-bottom: 15px;
    }
    .pred-class {
        font-size: 2rem;
        font-weight: bold;
        color: #2980B9;
        margin-bottom: 5px;
    }
    .pred-conf {
        font-size: 1.2rem;
        color: #34495E;
    }
    
    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Cached Model Initialization
# ==============================================================================
@st.cache_resource(show_spinner="Loading Model Architecture & Weights...")
def init_system():
    device = config.DEVICE
    model_path = os.path.join(config.MODEL_DIR, "best_model_optimized.pt")
    
    # If the model doesn't exist yet (e.g. training is still running), 
    # we fail gracefully so the app still launches.
    if not os.path.exists(model_path):
        return None, None, device
        
    try:
        model = load_inference_model(model_path, device)
        idx_to_class = load_class_mapping()
        return model, idx_to_class, device
    except Exception as e:
        # Catch architecture mismatch (state_dict runtime errors) from old runs
        st.warning(f"Model checkpoint found, but it is incompatible with the current architecture. Waiting for the new training run to overwrite it.")
        return None, None, device

# ==============================================================================
# Main Application
# ==============================================================================
def main():
    st.markdown('<div class="main-header">Skin Cancer Multi-Class Classification AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-assisted dermoscopic lesion classification</div>', unsafe_allow_html=True)

    model, idx_to_class, device = init_system()

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Settings")
        use_tta = st.toggle(
            "Enable Test Time Augmentation (TTA)", 
            value=False, 
            help="TTA improves accuracy by running inference on multiple augmented versions (flips, rotations) of the image and averaging. It takes slightly longer."
        )
        
        st.markdown("---")
        st.markdown("### About the Model")
        st.info(
            "**Architecture**: EfficientNet-B3 (12M Parameters)\n\n"
            "**Input Resolution**: 300x300 pixels\n\n"
            "**Classifier Target**: 9 highly granular skin cancer & lesion classes."
        )

    # If model is missing, show an alert and stop execution
    if model is None:
        st.error("⚠️ **Model Not Found**: The application could not locate `best_model_optimized.pt` in the outputs folder. The training script may still be running. Please wait for it to finish and refresh the page.")
        st.stop()

    # --- Main Layout ---
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("1. Upload Image")
        st.markdown("Please upload a clear dermoscopic image of the skin lesion.")
        uploaded_file = st.file_uploader("Choose a JPG or PNG file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image Preview", use_container_width=True)

            # Save temporarily for the inference script
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            temp_path = os.path.join(config.OUTPUT_DIR, "temp_upload.jpg")
            image.save(temp_path)

    with col2:
        if uploaded_file is not None:
            st.subheader("2. Prediction Results")
            
            with st.spinner("Analyzing microscopic lesion features..."):
                start_time = time.time()
                try:
                    # Run Inference internally, passing the cached model
                    pred_class, confidence, prob_dict = run_inference(
                        image_path=temp_path, 
                        use_tta=use_tta, 
                        model=model, 
                        idx_to_class=idx_to_class
                    )
                    inference_time = time.time() - start_time
                    
                    # Highlight Top Prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="pred-class">{pred_class.upper()}</div>
                        <div class="pred-conf">Confidence Score: <b>{confidence*100:.2f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption(f"⏱️ Inference completed in {inference_time:.2f} seconds")
                    
                    # Probability Distribution Chart
                    st.markdown("#### Probability Distribution")
                    df_probs = pd.DataFrame(list(prob_dict.items()), columns=['Class', 'Probability (%)'])
                    df_probs = df_probs.sort_values(by='Probability (%)', ascending=True)
                    
                    fig = px.bar(
                        df_probs, 
                        x='Probability (%)', 
                        y='Class', 
                        orientation='h',
                        color='Probability (%)',
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="Confidence (%)",
                        yaxis_title="",
                        height=350,
                        coloraxis_showscale=False,
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    # Force x-axis to run from 0 to 100 for proper scale perception
                    fig.update_xaxes(range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.stop()
        else:
            st.info("👈 Please upload an image in the left panel to see the AI prediction results.")

    # --- Grad-CAM Section ---
    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("3. Grad-CAM Analysis")
        st.markdown("*Highlighted regions (in red/yellow) indicate the exact morphological features influencing the model's ultimate decision.*")
        
        gradcam_path = os.path.join(config.GRADCAM_DIR, "inference_gradcam.png")
        if os.path.exists(gradcam_path):
            st.image(gradcam_path, use_container_width=True)
        else:
            st.warning("Grad-CAM visualization is currently unavailable for this prediction.")

    # --- Disclaimer Footer ---
    st.markdown("""
    <div class="disclaimer-box">
        <span class="disclaimer-text">⚠️ MEDICAL DISCLAIMER</span><br>
        This application represents an AI-based screening tool developed for portfolio and educational demonstration purposes only. 
        It is <b>not</b> a licensed medical diagnosis system. Deep Neural Networks can make mistakes. Always consult with a certified dermatologist or healthcare professional 
        for proper medical evaluation and advice regarding any skin conditions.
    </div>
    """, unsafe_allow_html=True)
            
if __name__ == "__main__":
    main()
