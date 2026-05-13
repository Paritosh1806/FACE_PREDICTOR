import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import cv2
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Face Collage Creator", layout="centered")

st.title("🖼️ Face Collage Creator")
st.write(
    "Upload a group photo or individual photo. "
    "The app detects human faces and creates a collage automatically."
)

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# Face Detection Function
# -----------------------------------
def detect_faces(image):
    img_np = np.array(image)

    # Convert RGB to BGR
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Detect Faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    face_images = []

    for (x, y, w, h) in faces:

        padding = int(0.2 * w)

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, img_cv.shape[1])
        y2 = min(y + h + padding, img_cv.shape[0])

        face = image.crop((x1, y1, x2, y2))

        face_images.append(face)

    return face_images, faces


# -----------------------------------
# Create Collage Function
# -----------------------------------
def create_collage(face_images, size=(200, 200), cols=3):

    processed_faces = []

    for img in face_images:
        img = ImageOps.fit(img, size)
        processed_faces.append(img)

    rows = (len(processed_faces) + cols - 1) // cols

    collage_width = cols * size[0]
    collage_height = rows * size[1]

    collage = Image.new(
        "RGB",
        (collage_width, collage_height),
        color="white"
    )

    for index, face in enumerate(processed_faces):

        x = (index % cols) * size[0]
        y = (index // cols) * size[1]

        collage.paste(face, (x, y))

    return collage


# -----------------------------------
# Main App
# -----------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Detecting faces..."):
        face_images, faces = detect_faces(image)

    # Draw rectangles around faces
    preview = image.copy()
    draw = ImageDraw.Draw(preview)

    for (x, y, w, h) in faces:
        draw.rectangle(
            (x, y, x + w, y + h),
            outline="red",
            width=4
        )

    st.image(
        preview,
        caption=f"Detected Faces: {len(face_images)}",
        use_container_width=True
    )

    if len(face_images) == 0:
        st.warning("No human faces detected.")

    else:

        st.success(f"{len(face_images)} face(s) detected.")

        if st.button("Create Collage"):

            collage = create_collage(face_images)

            st.image(
                collage,
                caption="Generated Face Collage",
                use_container_width=True
            )

            # -----------------------------------
            # Download JPG
            # -----------------------------------
            jpg_buffer = BytesIO()

            collage.save(jpg_buffer, format="JPEG")

            st.download_button(
                label="⬇ Download as JPG",
                data=jpg_buffer.getvalue(),
                file_name="face_collage.jpg",
                mime="image/jpeg"
            )

            # -----------------------------------
            # Download PDF
            # -----------------------------------
            pdf_buffer = BytesIO()

            collage_rgb = collage.convert("RGB")

            collage_rgb.save(pdf_buffer, format="PDF")

            st.download_button(
                label="⬇ Download as PDF",
                data=pdf_buffer.getvalue(),
                file_name="face_collage.pdf",
                mime="application/pdf"
            )
