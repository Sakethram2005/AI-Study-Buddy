import streamlit as st
import easyocr
from PIL import Image
import numpy as np
import cv2
import re
import fitz
from google import genai


API_KEY = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=API_KEY)


reader = easyocr.Reader(['en'])


def preprocess_image(img):

    img = np.array(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    return thresh


st.title("📘 AI Study Buddy ")

st.write("Upload handwritten or typed notes (Image or PDF)")

uploaded_file = st.file_uploader(
    "Upload notes",
    type=["png","jpg","jpeg","pdf"]
)

if uploaded_file is not None:

    extracted_text = ""

    if uploaded_file.type != "application/pdf":

        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        processed = preprocess_image(image)

        with st.spinner("Extracting text..."):

            result = reader.readtext(
                processed,
                detail=0,
                paragraph=True
            )

            extracted_text = " ".join(result)


    else:

        st.info("PDF detected. Processing pages...")

        pdf_bytes = uploaded_file.read()

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_num in range(len(doc)):

            page = doc.load_page(page_num)

            
            pix = page.get_pixmap(matrix=fitz.Matrix(3,3))

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            st.image(img, caption=f"Page {page_num+1}", use_column_width=True)

            processed = preprocess_image(img)

            with st.spinner(f"Reading page {page_num+1}"):

                result = reader.readtext(
                    processed,
                    detail=0,
                    paragraph=True
                )

                extracted_text += " ".join(result) + " "

    
    clean_text = re.sub(
        r'[^a-zA-Z0-9.,()+\-*/= ]',
        ' ',
        extracted_text
    )

    st.subheader("Extracted Text")

    st.write(clean_text)


    task = st.selectbox(
        "What should I do?",
        [
            "Summarize",
            "Explain Simply",
            "Generate Quiz"
        ]
    )

    if st.button("Run AI Buddy"):

        prompt = f"""
        Task: {task}

        Content:
        {clean_text}

        If the content contains mathematics,
        explain it clearly step-by-step.
        """

        with st.spinner("AI is thinking..."):

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

        st.success("Done! Here is your result")


        st.write(response.text)
