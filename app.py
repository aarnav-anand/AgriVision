import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(
    page_title="AgriVision - Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disease-name {
        font-size: 2rem;
        color: #1976D2;
        font-weight: bold;
    }
    .confidence-score {
        font-size: 1.5rem;
        color: #388E3C;
    }
    .treatment-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_classes():
    try:
        interpreter = tf.lite.Interpreter(model_path='plant_disease_model.tflite')
        interpreter.allocate_tensors()
        
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        
        index_to_class = {v: k for k, v in class_indices.items()}
        
        with open('treatment_database.json', 'r') as f:
            treatment_db = json.load(f)
        
        return interpreter, index_to_class, treatment_db
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def predict_disease(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    processed_image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    return predictions[0]

def get_treatment_info(class_name, treatment_db):
    if class_name in treatment_db:
        return treatment_db[class_name]
    else:
        return treatment_db.get('default', {
            'disease_name': 'Unknown',
            'description': 'No information available',
            'organic_treatment': ['Consult an expert'],
            'chemical_treatment': ['Consult an expert'],
            'prevention': ['Regular monitoring']
        })

def main():
    st.markdown('<h1 class="main-header">🌿 AgriVision</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Plant Disease Detection for Small-Scale Farmers</p>', unsafe_allow_html=True)
    
    interpreter, index_to_class, treatment_db = load_model_and_classes()
    
    if interpreter is None:
        st.error("❌ Failed to load model. Please check your model files.")
        return
    
    with st.sidebar:
        st.header("📋 About AgriVision")
        st.write("""
        **AgriVision** uses advanced AI to identify plant diseases from photos, 
        helping farmers make informed decisions about crop treatment.
        
        **How to use:**
        1. Upload a clear photo of a diseased plant leaf
        2. Get instant disease diagnosis
        3. Receive treatment recommendations
        
        **Supported crops:** Apple, Tomato, Potato, Corn, Grape, and more!
        """)
        
        st.header("📊 Model Info")
        try:
            with open('models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
            st.write(f"**Accuracy:** {metadata.get('test_accuracy', 0)*100:.1f}%")
            st.write(f"**Classes:** {metadata.get('num_classes', 'N/A')}")
            st.write(f"**Model Size:** {metadata.get('model_size_tflite_mb', 'N/A')} MB")
        except:
            st.write("Metadata not available")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Plant Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image of a plant leaf",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear, well-lit photo of a single leaf"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button("🔍 Analyze Disease", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    predictions = predict_disease(interpreter, image)
                    
                    predicted_class_idx = np.argmax(predictions)
                    confidence = predictions[predicted_class_idx]
                    predicted_class_name = index_to_class[predicted_class_idx]
                    
                    top_3_indices = np.argsort(predictions)[-3:][::-1]
                    
                    st.session_state['predictions'] = predictions
                    st.session_state['predicted_class_name'] = predicted_class_name
                    st.session_state['confidence'] = confidence
                    st.session_state['top_3_indices'] = top_3_indices
        
        else:
            st.info("👆 Please upload an image to get started")
            
            st.subheader("💡 Tips for Best Results")
            st.write("""
            - Use a clear, focused image
            - Ensure good lighting
            - Capture the entire leaf
            - Avoid blurry or dark photos
            - Take photo against a plain background
            """)
    
    with col2:
        st.header("🔬 Diagnosis Results")
        
        if 'predicted_class_name' in st.session_state:
            predicted_class_name = st.session_state['predicted_class_name']
            confidence = st.session_state['confidence']
            predictions = st.session_state['predictions']
            top_3_indices = st.session_state['top_3_indices']
            
            treatment_info = get_treatment_info(predicted_class_name, treatment_db)
            
            st.markdown(f'<p class="disease-name">🦠 {treatment_info["disease_name"]}</p>', unsafe_allow_html=True)
            
            confidence_pct = confidence * 100
            if confidence_pct >= 80:
                conf_color = "green"
            elif confidence_pct >= 60:
                conf_color = "orange"
            else:
                conf_color = "red"
            
            st.markdown(f'<p class="confidence-score">Confidence: <span style="color:{conf_color}">{confidence_pct:.1f}%</span></p>', unsafe_allow_html=True)
            
            st.write(f"**Description:** {treatment_info['description']}")
            
            with st.expander("📊 View Top 3 Predictions"):
                for idx in top_3_indices:
                    class_name = index_to_class[idx]
                    prob = predictions[idx] * 100
                    st.write(f"**{class_name}:** {prob:.1f}%")
            
            st.markdown('<div class="treatment-section">', unsafe_allow_html=True)
            st.subheader("💊 Treatment Recommendations")
            
            tab1, tab2, tab3 = st.tabs(["🌱 Organic", "🧪 Chemical", "🛡️ Prevention"])
            
            with tab1:
                st.write("**Organic Treatment Options:**")
                for i, treatment in enumerate(treatment_info['organic_treatment'], 1):
                    st.write(f"{i}. {treatment}")
            
            with tab2:
                st.write("**Chemical Treatment Options:**")
                for i, treatment in enumerate(treatment_info['chemical_treatment'], 1):
                    st.write(f"{i}. {treatment}")
            
            with tab3:
                st.write("**Prevention Measures:**")
                for i, prevention in enumerate(treatment_info['prevention'], 1):
                    st.write(f"{i}. {prevention}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if confidence_pct < 70:
                st.warning("⚠️ Low confidence detection. Please consult an agricultural expert for confirmation.")
            
            if st.button("📄 Download Report", use_container_width=True):
                report = f"""
AGRIVISION DISEASE DIAGNOSIS REPORT
====================================

Disease: {treatment_info['disease_name']}
Confidence: {confidence_pct:.1f}%

Description:
{treatment_info['description']}

ORGANIC TREATMENT:
{chr(10).join([f"{i+1}. {t}" for i, t in enumerate(treatment_info['organic_treatment'])])}

CHEMICAL TREATMENT:
{chr(10).join([f"{i+1}. {t}" for i, t in enumerate(treatment_info['chemical_treatment'])])}

PREVENTION:
{chr(10).join([f"{i+1}. {p}" for i, p in enumerate(treatment_info['prevention'])])}

---
Generated by AgriVision - AI Plant Disease Detection
"""
                st.download_button(
                    label="Download Text Report",
                    data=report,
                    file_name="agrivision_report.txt",
                    mime="text/plain"
                )
        
        else:
            st.info("👈 Upload an image and click 'Analyze' to see results")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌾 AgriVision - Democratizing Agricultural Diagnostics | Powered by AI</p>
        <p>⚠️ Disclaimer: This tool provides guidance only. For critical decisions, consult agricultural experts.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()

