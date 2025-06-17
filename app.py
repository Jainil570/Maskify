import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import load_model
from PIL import Image
import io

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 128, 128, 3
MODEL_PATH = r'C:\Users\hp\Desktop\Jainil\image_seg\Human-Segmentation-Dataset-master\model_epoch32_val0.2406.h5'

@st.cache_resource
def load_segmentation_model():
    return load_model(MODEL_PATH)

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    original_size = image.size
    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image_resized)  
    return image, np.expand_dims(image_array, axis=0), original_size

def remove_background(original_image, prediction, original_size):
    mask = (prediction[0].squeeze() > 0.5).astype(np.uint8)
    
    mask_resized = Image.fromarray(mask * 255).resize(original_size)
    mask_resized_np = np.array(mask_resized)

    image_rgba = original_image.convert("RGBA")
    pixels = np.array(image_rgba)

    pixels[..., 3] = mask_resized_np  
    return Image.fromarray(pixels)

st.title("🔍 Background Remover using Segmentation")
st.write("Upload an image to remove its background using a U-Net model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    model = load_segmentation_model()
    original_img, input_tensor, original_size = preprocess_image(uploaded_file)

    st.image(original_img, caption='Original Image', use_container_width=True)

    prediction = model.predict(input_tensor)

    st.subheader("Segmentation Mask Preview")
    st.image(prediction[0].squeeze(), caption="Raw Prediction Mask", use_container_width=True, clamp=True)

    output_img = remove_background(original_img, prediction, original_size)

    st.subheader("Final Output")
    st.image(output_img, caption='Background Removed', use_container_width=True)

    buf = io.BytesIO()
    output_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download Transparent PNG", data=byte_im, file_name="output.png", mime="image/png")
