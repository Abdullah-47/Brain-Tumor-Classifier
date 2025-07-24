# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import plotly.express as px

# Load saved model (custom CNN)
@st.cache_resource
def load_model():
    """Loads and returns the trained Keras model."""
    model = tf.keras.models.load_model('best_model.h5')
    return model

model = load_model()

# Class labels with descriptions
tumor_info = {
    'glioma': {
        'description': "Glioma is a type of tumor that occurs in the brain and spinal cell. Gliomas begin in the gluey supportive cells (glial cells) that surround nerve cells.",
        'prevalence': "Most common malignant brain tumor in adults",
        'treatment': "Surgery, radiation therapy, chemotherapy"
    },
    'meningioma': {
        'description': "Meningioma is a tumor that arises from the meninges — the membranes that surround the brain and spinal cord. Most meningiomas are non-cancerous (benign).",
        'prevalence': "Most common primary brain tumor (30% of all brain tumors)",
        'treatment': "Monitoring, surgery, radiation therapy"
    },
    'no_tumor': {
        'description': "No signs of tumor detected in the MRI scan. Normal brain tissue appears healthy.",
        'prevalence': "Normal brain MRI",
        'treatment': "No treatment needed"
    },
    'pituitary': {
        'description': "Pituitary tumors are abnormal growths that develop in the pituitary gland. Most are benign and many don't cause symptoms.",
        'prevalence': "10-15% of all primary brain tumors",
        'treatment': "Medication, surgery, radiation therapy"
    }
}

def generate_gradcam(model, img_array, interpolant=0.5):
    """
    Generates Grad-CAM visualization for a custom CNN model.
    Args:
        model: Compiled Keras model.
        img_array: Preprocessed image array (1, 224, 224, 3).
        interpolant: Opacity for heatmap overlay (0-1).
    Returns:
        tuple: (superimposed_img, heatmap) or (None, error_message).
    """
    try:
        # Find the last convolutional layer automatically
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        if last_conv_layer is None:
            raise ValueError("No Conv2D layer found in the model.")

        # Define a symbolic input tensor for the new `gradient_model`.
        grad_model_input = tf.keras.Input(shape=img_array.shape[1:])

        # Reconstruct the forward pass *symbolically* through the original model's layers
        x = grad_model_input
        last_conv_output_symbolic = None
        
        for layer in model.layers:
            x = layer(x)
            if layer == last_conv_layer:
                last_conv_output_symbolic = x

        final_output_symbolic = x

        if last_conv_output_symbolic is None:
            raise ValueError(f"Could not find the symbolic output for the last convolutional layer ('{last_conv_layer.name}').")

        gradient_model = tf.keras.models.Model(
            inputs=grad_model_input,
            outputs=[last_conv_output_symbolic, final_output_symbolic]
        )

        with tf.GradientTape() as tape:
            inputs_for_tape = tf.cast(img_array, tf.float32)
            conv_outputs, predictions = gradient_model(inputs_for_tape)
            
            # Use argmax to get the predicted class index
            pred_index = tf.argmax(predictions[0])
            
            # Extract the loss for the predicted class
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        
        # --- Crucial Error Handling for Gradients & Heatmap ---
        if grads is None:
            return None, "Grad-CAM failed: Gradients are None. This might indicate an issue with differentiability or an unusual model state for this input."

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs by pooled gradients
        conv_outputs = conv_outputs[0] # Remove batch dimension
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) # Apply ReLU to heatmap
        
        # Check if heatmap is all zeros AFTER ReLU. If so, normalization will fail.
        max_heatmap_value = tf.math.reduce_max(heatmap)
        if tf.equal(max_heatmap_value, 0):
            return None, "Grad-CAM failed: Heatmap is entirely zero, cannot normalize. This may happen if the model's activations or gradients are all zero for this input and predicted class."

        heatmap = heatmap / max_heatmap_value # Normalize by max value
        heatmap = heatmap.numpy()
        
        # Resize heatmap to original image size
        heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
        
        # Convert to RGB heatmap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Prepare original image
        img = np.uint8(img_array[0] * 255)
        
        # Superimpose heatmap on original image
        superimposed_img = cv2.addWeighted(
            img, 1 - interpolant,
            heatmap_colored, interpolant,
            0
        )
        
        return superimposed_img, heatmap
    
    except Exception as e:
        return None, f"Grad-CAM failed: {str(e)}"

# --- Streamlit UI (No changes needed below this point) ---
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .sidebar .sidebar-content {
        background: #0c151c !important;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3d8b40;
        transform: scale(1.05);
    }
    .prediction-highlight {
        font-size: 28px;
        font-weight: bold;
        color: #4CAF50;
        text-shadow: 0 0 8px rgba(76, 175, 80, 0.4);
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 136, 229, 0.2) !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background: #1e88e5 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🧠 Brain Tumor MRI Classification")
st.markdown("This AI-powered tool analyzes brain MRI scans to detect and classify tumors using a Convolutional Neural Network (CNN). Upload an MRI image to get a prediction and detailed insights.")

# Sidebar with info and metrics
with st.sidebar:
    st.header("Model Information")
    st.markdown("""
    - **Model Architecture**: Custom CNN
    - **Training Data**: 1,695 MRI scans
    - **Test Accuracy**: 76.0%
    - **Balanced Accuracy**: 74.8%
    - **Macro F1-Score**: 74.5%
    """)
    st.divider()
    st.header("Performance by Tumor Type")
    with st.expander("Glioma"):
        st.markdown("**Precision**: 0.78 | **Recall**: 0.93 | **F1-Score**: 0.85")
    with st.expander("Meningioma"):
        st.markdown("**Precision**: 0.65 | **Recall**: 0.51 | **F1-Score**: 0.57")
    with st.expander("No Tumor"):
        st.markdown("**Precision**: 0.89 | **Recall**: 0.63 | **F1-Score**: 0.74")
    with st.expander("Pituitary"):
        st.markdown("**Precision**: 0.75 | **Recall**: 0.93 | **F1-Score**: 0.83")
    st.divider()
    st.warning("⚠️ **Disclaimer**: This tool is for educational purposes only. Always consult a medical professional for diagnosis.")

# Main content area
col1, col2 = st.columns([1, 1])

if 'prediction_made' not in st.session_state:
    st.session_state['prediction_made'] = False
    st.session_state['predicted_class'] = None
    st.session_state['confidence'] = None
    st.session_state['gradcam_img'] = None
    st.session_state['heatmap_error'] = None
    st.session_state['prediction_probs'] = None


with col1:
    st.subheader("Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image (JPEG)", 
        type=["jpg","jpeg"],
        label_visibility="collapsed"
    )

if uploaded_file is not None:
    # --- Single Analysis Block ---
    image = Image.open(uploaded_file).convert('RGB')
    uploaded_file.close()
    img_display = image.copy()
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Display uploaded image in the second column
    with col2:
        st.subheader("Uploaded MRI Scan")
        st.image(img_display, caption="Original MRI", use_container_width=True)
    
    with st.spinner('Analyzing MRI scan...'):
        prediction = model.predict(img_array, verbose=0) 
        
        predicted_class = list(tumor_info.keys())[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        gradcam_img, heatmap_status = generate_gradcam(model, img_array, interpolant=0.6)

        st.session_state['prediction_made'] = True
        st.session_state['predicted_class'] = predicted_class
        st.session_state['confidence'] = confidence
        st.session_state['gradcam_img'] = gradcam_img
        st.session_state['heatmap_error'] = heatmap_status
        st.session_state['prediction_probs'] = prediction[0]

# --- Results Section (display only if prediction was made) ---
if st.session_state['prediction_made']:
    st.divider()
    
    col_res1, col_res2 = st.columns([1, 2])
      
    with col_res1:
        st.subheader("AI Analysis Result")
        st.markdown(f"<div class='prediction-highlight'>{st.session_state['predicted_class'].replace('_', ' ').title()}</div>", unsafe_allow_html=True)
        
        st.metric("Confidence Level", f"{st.session_state['confidence']:.2f}%")
        st.progress(int(st.session_state['confidence']))
        
        info = tumor_info[st.session_state['predicted_class']]
        with st.expander("Tumor Information", expanded=True):
            st.markdown(f"**Description**: {info['description']}")
            st.markdown(f"**Prevalence**: {info['prevalence']}")
            st.markdown(f"**Treatment**: {info['treatment']}")
        
        st.info("💡 **Clinical Note**: The AI analysis should be reviewed by a qualified radiologist. It is not a substitute for professional medical diagnosis.")

    with col_res2:
        st.subheader("Model Insights")
        
        tab1, tab2, tab3 = st.tabs(["📊 Probability Distribution", "🔥 Attention Map", "📈 Performance Metrics"])
        
        with tab1:
            classes = list(tumor_info.keys())
            probs = st.session_state['prediction_probs'] * 100
            
            # Use Plotly instead of matplotlib
            fig = px.bar(
                x=probs,
                y=[c.replace('_', ' ').title() for c in classes],
                orientation='h',
                text=probs,
                color=[c.replace('_', ' ').title() for c in classes],
                color_discrete_sequence=['#1e88e5' if c != st.session_state['predicted_class'] else '#4CAF50' for c in classes]
            )
            fig.update_layout(
                title='Prediction Confidence Distribution',
                xaxis_title='Probability (%)',
                yaxis_title='Class',
                showlegend=False,
                uniformtext_minsize=12,
                uniformtext_mode='hide'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if st.session_state['gradcam_img'] is not None:
                st.image(st.session_state['gradcam_img'], caption="AI Attention Map (Grad-CAM)", use_container_width=True)
                st.markdown("The highlighted areas indicate the regions the model focused on to make its prediction.")
            else:
                st.warning(st.session_state['heatmap_error']) # Display error message from generate_gradcam
        
        with tab3:
            cnn_cm = np.array([
                [74, 6, 0, 0],    # glioma
                [17, 32, 4, 10],  # meningioma
                [1, 10, 31, 7],   # no_tumor
                [3, 1, 0, 50]     # pituitary
            ])
            
            st.write("**Custom CNN Confusion Matrix (Test Set)**")
            
            # FIX: Create a container and use st.plotly_chart for stable rendering
            cm_container = st.container()
            
            # Use Plotly for stable visualization
            fig = px.imshow(
                cnn_cm,
                text_auto=True,
                labels=dict(x="Predicted", y="True", color="Count"),
                x=[c.replace('_', ' ').title() for c in tumor_info.keys()],
                y=[c.replace('_', ' ').title() for c in tumor_info.keys()],
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                autosize=True
            )
            cm_container.plotly_chart(fig, use_container_width=True)
            
            st.write("**Performance by Class:**")
            class_data = {
                'Tumor Type': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
                'Precision': [0.78, 0.65, 0.89, 0.75],
                'Recall': [0.93, 0.51, 0.63, 0.93],
                'F1-Score': [0.85, 0.57, 0.74, 0.83]
            }
            df = pd.DataFrame(class_data).set_index('Tumor Type')
            st.dataframe(df.style.format("{:.2f}").highlight_max(axis=0, color='rgba(76, 175, 80, 0.3)'))
# Footer
st.markdown("---")
st.caption("© 2025 Brain Tumor Classification System | For Research Use Only")