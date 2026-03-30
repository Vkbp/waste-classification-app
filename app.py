
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Phân Loại Rác Thải", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('waste_classification_model.h5')

model = load_model()

classes = ['battery', 'glass', 'metal', 'organic', 'paper', 'plastic']

tips = {
    'battery': '🔋 Pin - Đưa đến điểm thu gom pin thải nguy hại. KHÔNG bỏ chung rác thường.',
    'glass': '🪟 Thủy tinh - Rửa sạch, bỏ vào thùng tái chế thủy tinh.',
    'metal': '🔩 Kim loại - Bỏ vào thùng tái chế kim loại.',
    'organic': '🍃 Hữu cơ - Bỏ vào thùng rác hữu cơ để làm phân compost.',
    'paper': '📄 Giấy - Bỏ vào thùng tái chế giấy.',
    'plastic': '♻️ Nhựa - Phân loại theo số (1-7), bỏ vào thùng tái chế nhựa.'
}

st.title("♻️ Phân Loại Rác Thải Thông Minh")

st.markdown("**Upload ảnh hoặc chụp trực tiếp** để hệ thống dự đoán loại rác.")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("📤 Upload ảnh", type=["jpg", "jpeg", "png"])
with col2:
    camera = st.camera_input("📸 Chụp trực tiếp")

img = None
if uploaded_file is not None:
    img = Image.open(uploaded_file)
elif camera is not None:
    img = Image.open(camera)

if img is not None:
    st.image(img, caption="Ảnh rác", use_column_width=True)
    
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(prediction)
    pred_class = classes[pred_idx]
    confidence = prediction[0][pred_idx] * 100
    
    st.success(f"**Loại rác:** {pred_class.upper()} ({confidence:.2f}%)")
    st.info(f"**Hướng dẫn xử lý:** {tips[pred_class]}")
    
    st.subheader("Xác suất chi tiết:")
    for i, cls in enumerate(classes):
        st.write(f"• {cls.capitalize()}: **{prediction[0][i]*100:.2f}%**")
