import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import plotly.express as px
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model download link (Google Drive)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"
MODEL_PATH = "Modelenv.v1.h5"

# Download model if not present
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... (only once)"):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                return None
    
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Class labels and their descriptions
class_info = {
    'Cloudy': {
        'emoji': '☁️',
        'description': 'Areas covered by clouds or atmospheric conditions',
        'color': '#87CEEB'
    },
    'Desert': {
        'emoji': '🏜️',
        'description': 'Arid, sandy, or rocky terrain with minimal vegetation',
        'color': '#F4A460'
    },
    'Green_Area': {
        'emoji': '🌿',
        'description': 'Vegetation, forests, agricultural land, or grasslands',
        'color': '#90EE90'
    },
    'Water': {
        'emoji': '💧',
        'description': 'Bodies of water including lakes, rivers, and oceans',
        'color': '#4682B4'
    }
}

class_names = list(class_info.keys())

# Main UI
st.title("🌍 Satellite Image Classifier")
st.markdown("""
<div style="background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
    <h4>🛰️ About This App</h4>
    <p>This AI-powered application classifies satellite images into four categories: Cloudy, Desert, Green Area, and Water. 
    Upload a satellite image and get instant predictions with confidence scores!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 Classification Categories")
    for class_name, info in class_info.items():
        st.markdown(f"""
        <div style="background-color: {info['color']}; padding: 0.5rem; margin: 0.5rem 0; border-radius: 5px; opacity: 0.7;">
            <strong>{info['emoji']} {class_name}</strong><br>
            <small>{info['description']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info("Input Size: 256x256 pixels\nModel: CNN (Keras/TensorFlow)")

# Load model
model = download_and_load_model()

if model is None:
    st.error("❌ Failed to load the model. Please check your internet connection and try again.")
    st.stop()

# File uploader
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload a satellite image", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

with col2:
    if st.button("🔄 Clear Results"):
        st.rerun()

# Process uploaded image
if uploaded_file is not None:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert("RGB")
        original_size = image.size
        image_resized = image.resize((256, 256))
        
        # Display image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, caption=f"Original Size: {original_size[0]}x{original_size[1]}", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Processed Image")
            st.image(image_resized, caption="Resized to 256x256", use_container_width=True)
        
        # Preprocess for prediction
        img_array = img_to_array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        with st.spinner("🤖 Analyzing image..."):
            prediction = model.predict(img_array)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)
        
        # Display results
        st.markdown("---")
        st.subheader("🎯 Classification Results")
        
        # Main prediction
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div style="background-color: {class_info[predicted_class]['color']}; padding: 1rem; border-radius: 10px; text-align: center;">
                <h2>{class_info[predicted_class]['emoji']}</h2>
                <h3>{predicted_class}</h3>
                <p><strong>Confidence: {confidence * 100:.2f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Confidence meter
            st.metric(
                label="Confidence Level",
                value=f"{confidence * 100:.2f}%",
                delta=f"{'High' if confidence > 0.8 else 'Medium' if confidence > 0.5 else 'Low'} confidence"
            )
        
        with col3:
            # Category description
            st.info(f"**{predicted_class}**: {class_info[predicted_class]['description']}")
        
        # Detailed predictions chart
        st.subheader("📊 All Class Probabilities")
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'Class': class_names,
            'Probability': prediction * 100,
            'Emoji': [class_info[cls]['emoji'] for cls in class_names]
        })
        
        # Create bar chart
        fig = px.bar(
            df, 
            x='Class', 
            y='Probability',
            color='Probability',
            color_continuous_scale='viridis',
            title="Classification Probabilities for All Classes"
        )
        fig.update_layout(
            xaxis_title="Class",
            yaxis_title="Probability (%)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown
        st.subheader("🔢 Detailed Breakdown")
        for i, (class_name, prob) in enumerate(zip(class_names, prediction)):
            emoji = class_info[class_name]['emoji']
            is_predicted = class_name == predicted_class
            
            if is_predicted:
                st.success(f"{emoji} **{class_name}**: {prob*100:.2f}% ← **PREDICTED**")
            else:
                st.write(f"{emoji} **{class_name}**: {prob*100:.2f}%")
        
        # Additional info
        st.markdown("---")
        st.markdown("### 💡 Tips for Better Results")
        st.markdown("""
        - Use high-quality satellite images
        - Ensure the image clearly shows the terrain type
        - Images with mixed terrain types may have lower confidence
        - Best results with images similar to training data
        """)
        
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        st.info("Please try uploading a different image or check if the file is corrupted.")

else:
    # Default state
    st.info("👆 Please upload a satellite image to get started!")
    
    # Example images section
    st.markdown("---")
    st.subheader("🖼️ Example Classifications")
    
    example_cols = st.columns(4)
    for i, (class_name, info) in enumerate(class_info.items()):
        with example_cols[i]:
            st.markdown(f"""
            <div style="background-color: {info['color']}; padding: 1rem; border-radius: 10px; text-align: center; opacity: 0.8;">
                <h2>{info['emoji']}</h2>
                <h4>{class_name}</h4>
                <p style="font-size: 0.8em;">{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🚀 Built with Streamlit & TensorFlow | 🤖 AI-Powered Satellite Image Classification</p>
</div>
""", unsafe_allow_html=True)