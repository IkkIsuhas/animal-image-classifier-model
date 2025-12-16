import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🦁",
    layout="wide"
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer',
    'Dog', 'Dolphin', 'Elephant', 'Giraffe',
    'Horse', 'Kangaroo', 'Lion', 'Panda',
    'Tiger', 'Zebra'
]

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@st.cache_resource
def load_model():
    """Load the trained model"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load("animal_resnet.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image):
    """Predict the animal in the image"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Transform the image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
        
        # Get probabilities
        probabilities = torch.softmax(output, dim=1)
        predicted_index = probabilities.argmax(dim=1).item()
        confidence = probabilities[0][predicted_index].item() * 100
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        top3_predictions = [
            {
                'class': class_names[idx.item()],
                'confidence': prob.item() * 100
            }
            for prob, idx in zip(top3_probs, top3_indices)
        ]
        
        return {
            'predicted_class': class_names[predicted_index],
            'confidence': confidence,
            'top3': top3_predictions,
            'all_probs': probabilities[0].cpu().numpy()
        }
    except Exception as e:
        return {'error': str(e)}

# Load model
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model_loaded = False

# Main UI
st.title("🦁 Animal Classifier")
st.markdown("Upload an image of an animal and get predictions!")

# Sidebar
with st.sidebar:
    st.header("📋 Supported Animals")
    st.markdown("""
    - 🐻 Bear
    - 🐦 Bird
    - 🐱 Cat
    - 🐄 Cow
    - 🦌 Deer
    - 🐕 Dog
    - 🐬 Dolphin
    - 🐘 Elephant
    - 🦒 Giraffe
    - 🐴 Horse
    - 🦘 Kangaroo
    - 🦁 Lion
    - 🐼 Panda
    - 🐅 Tiger
    - 🦓 Zebra
    """)
    
    st.markdown("---")
    st.info(f"Using device: **{device}**")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of an animal"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Predict button
        if st.button("🔍 Predict Animal", type="primary", use_container_width=True):
            if model_loaded:
                with st.spinner("Analyzing image..."):
                    result = predict_image(model, image)
                
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Display results in col2
                    with col2:
                        st.header("🎯 Prediction Results")
                        
                        # Main prediction
                        st.success(f"**Predicted Animal: {result['predicted_class']}**")
                        st.metric("Confidence", f"{result['confidence']:.2f}%")
                        
                        # Progress bar for confidence
                        st.progress(result['confidence'] / 100)
                        
                        # Top 3 predictions
                        st.subheader("🏆 Top 3 Predictions")
                        for i, pred in enumerate(result['top3'], 1):
                            col_pred, col_conf = st.columns([2, 1])
                            with col_pred:
                                st.write(f"{i}. **{pred['class']}**")
                            with col_conf:
                                st.write(f"{pred['confidence']:.2f}%")
                            
                            # Progress bar for each prediction
                            st.progress(pred['confidence'] / 100)
                        
                        # All predictions chart
                        st.subheader("📊 All Predictions")
                        import pandas as pd
                        df = pd.DataFrame({
                            'Animal': class_names,
                            'Confidence (%)': result['all_probs'] * 100
                        })
                        df = df.sort_values('Confidence (%)', ascending=False)
                        st.bar_chart(df.set_index('Animal'))
            else:
                st.error("Model not loaded. Please check the error message above.")

with col2:
    if uploaded_file is None:
        st.info("👈 Upload an image to get started!")
        st.markdown("""
        ### How to use:
        1. Click "Browse files" or drag and drop an image
        2. Wait for the image to load
        3. Click "Predict Animal" button
        4. View the results!
        """)

