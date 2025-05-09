# app.py
import streamlit as st
from PIL import Image
from predict import load_model, predict, draw_boxes

st.set_page_config(page_title="BCCD Object Detection", layout="centered")
st.title("🔬 BCCD Object Detection (RBC, WBC, Platelets)")
st.write("Upload a microscope image to detect blood components.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model (cached)
    @st.cache_resource
    def get_model():
        return load_model()

    model = get_model()

    # Run prediction
    with st.spinner("Detecting cells..."):
        results = predict(image, model, threshold=0.5)
        image_with_boxes = draw_boxes(image.copy(), results)

    st.image(image_with_boxes, caption="Detection Results", use_column_width=True)

    # Display label summary
    st.subheader("Detected Cells")
    if results:
        for r in results:
            st.write(f"🩸 {r['label']} ({r['score']:.2f}) at {r['box']}")
    else:
        st.warning("No cells detected above threshold.")

