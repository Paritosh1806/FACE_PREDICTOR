import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import cv2
import numpy as np
from io import BytesIO
import urllib.request
import os

st.set_page_config(page_title="Face Collage Creator", layout="centered")

st.title("🖼️ AI Face Collage Creator")
st.write("Upload a group photo or selfie. The app detects ONLY human faces using AI.")

# -------------------------------------------------------
# Download DNN Model Files Automatically
# -------------------------------------------------------
MODEL_DIR = "models"

PROTO_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "res10_300x300_ssd_iter_140000.caffemodel"
)

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(PROTO_PATH):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        PROTO_PATH
    )

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/raw/master/res10_300x300_ssd_iter_140000.caffemodel",
        MODEL_PATH
    )

# Load DNN Model
net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------------
# Face Detection Function
# -------------------------------------------------------
def detect_faces_dnn(image):

    img_np = np.array(image)

    h, w = img_np.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_np, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)

    detections = net.forward()

    face_boxes = []

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # Confidence threshold
        if confidence > 0.6:

            box = detections[0, 0, i, 3:7] * np.array(
                [w, h, w, h]
            )

            (x1, y1, x2, y2) = box.astype("int")

            # Padding
            padding = int((x2 - x1) * 0.2)

            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            face_boxes.append((x1, y1, x2, y2))

    return face_boxes


# -------------------------------------------------------
# Create Collage
# -------------------------------------------------------
def create_collage(image, face_boxes):

    face_size = 220
    cols = 3

    face_images = []

    for (x1, y1, x2, y2) in face_boxes:

        face = image.crop((x1, y1, x2, y2))

        face = ImageOps.fit(
            face,
            (face_size, face_size)
        )

        face_images.append(face)

    rows = (len(face_images) + cols - 1) // cols

    collage_width = cols * face_size
    collage_height = rows * face_size

    collage = Image.new(
        "RGB",
        (collage_width, collage_height),
        color="white"
    )

    for index, face in enumerate(face_images):

        x = (index % cols) * face_size
        y = (index // cols) * face_size

        collage.paste(face, (x, y))

    return collage


# -------------------------------------------------------
# Main App
# -------------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Detecting faces using AI model..."):

        face_boxes = detect_faces_dnn(image)

    # Draw rectangles
    preview = image.copy()

    draw = ImageDraw.Draw(preview)

    for (x1, y1, x2, y2) in face_boxes:

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=4
        )

    st.image(
        preview,
        caption=f"Detected Faces: {len(face_boxes)}",
        use_container_width=True
    )

    if len(face_boxes) == 0:

        st.warning("No faces detected.")

    else:

        st.success(f"{len(face_boxes)} face(s) detected.")

        if st.button("Create Collage"):

            collage = create_collage(image, face_boxes)

            st.image(
                collage,
                caption="Generated Face Collage",
                use_container_width=True
            )

            # JPG Download
            jpg_buffer = BytesIO()

            collage.save(jpg_buffer, format="JPEG")

            st.download_button(
                label="⬇ Download JPG",
                data=jpg_buffer.getvalue(),
                file_name="face_collage.jpg",
                mime="image/jpeg"
            )

            # PDF Download
            pdf_buffer = BytesIO()

            collage_rgb = collage.convert("RGB")

            collage_rgb.save(pdf_buffer, format="PDF")

            st.download_button(
                label="⬇ Download PDF",
                data=pdf_buffer.getvalue(),
                file_name="face_collage.pdf",
                mime="application/pdf"
            )
