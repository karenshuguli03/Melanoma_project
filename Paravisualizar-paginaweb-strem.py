import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2

# CONFIGURATION

MODEL_PATH = "efficientnetb0_final.keras"
IMG_SIZE = (224, 224)
THRESHOLD = 0.3

# LOAD MODEL


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()

# IMAGE PREPROCESSING

def preprocess_image(image):
    image = image.convert("RGB")
    img_resized = image.resize(IMG_SIZE)
    img_array = keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    original_img = np.array(img_resized)

    return img_array, original_img

# GRAD-CAM FUNCTIONS

def find_last_4d_layer(base_model):
    for layer in reversed(base_model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except:
            pass

    raise ValueError("No valid 4D layer found for Grad-CAM.")

def make_gradcam_heatmap(img_array, model):
    base_model = model.get_layer("efficientnetb0")
    last_layer_name = find_last_4d_layer(base_model)

    base_grad_model = keras.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(last_layer_name).output,
            base_model.output
        ]
    )

    gap_layer = None
    dense_layer = None

    for layer in model.layers:
        if isinstance(layer, keras.layers.GlobalAveragePooling2D):
            gap_layer = layer
        if isinstance(layer, keras.layers.Dense):
            dense_layer = layer

    if gap_layer is None or dense_layer is None:
        raise ValueError("Final model layers were not found.")

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = base_grad_model(img_array, training=False)
        x = gap_layer(base_outputs)
        predictions = dense_layer(x)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    if max_value != 0:
        heatmap = heatmap / max_value

    return heatmap.numpy()

# GRAD-CAM + ROI VISUALIZATION

def create_gradcam_and_roi(original_img, heatmap, alpha=0.45):
    img_rgb = cv2.resize(original_img, IMG_SIZE)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Grad-CAM visualization
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    gradcam_img = cv2.addWeighted(
        heatmap_color,
        alpha,
        img_bgr,
        1 - alpha,
        0
    )

    # ROI detection based on dark lesion areas
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    _, dark_mask = cv2.threshold(
        l_channel,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    roi_img = img_bgr.copy()
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 150:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h != 0 else 0

            # Avoid very thin regions such as hair
            if 0.3 < aspect_ratio < 3.5:
                valid_contours.append(contour)

    if valid_contours:
        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)

        padding = 8
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, IMG_SIZE[0])
        y2 = min(y + h + padding, IMG_SIZE[1])

        cv2.rectangle(
            roi_img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

    gradcam_rgb = cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB)
    roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)

    return gradcam_rgb, roi_rgb

# APP UI

st.title("DermaScan AI")
st.write("AI-based tool for melanoma detection using skin lesion images.")

st.subheader("Capture or upload an image")

camera_image = st.camera_input("Take a photo")

uploaded_file = st.file_uploader(
    "Or upload an image",
    type=["jpg", "jpeg", "png"]
)

image_source = camera_image if camera_image is not None else uploaded_file
# MAIN PROCESS

if image_source is not None:
    image = Image.open(image_source).convert("RGB")

    img_array, original_img = preprocess_image(image)

    pred = model.predict(img_array)[0][0]

    if pred > THRESHOLD:
        label = "Malignant"
        risk = "High risk"
    else:
        label = "Benign"
        risk = "Low risk"

    st.subheader("Prediction Results")

    st.write(f"**Prediction:** {label}")
    st.write(f"**Malignancy probability:** {pred:.2%}")
    st.write(f"**Risk level:** {risk}")

    heatmap = make_gradcam_heatmap(img_array, model)
    gradcam_img, roi_img = create_gradcam_and_roi(original_img, heatmap)

    st.subheader("Visual Explanation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original_img, caption="Original Image", use_container_width=True)

    with col2:
        st.image(gradcam_img, caption="Grad-CAM Heatmap", use_container_width=True)

    with col3:
        st.image(roi_img, caption="Estimated ROI", use_container_width=True)

    st.warning(
        "This result is only an AI-based support tool and does NOT replace professional medical diagnosis."
    )