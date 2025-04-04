# data_processing.py
import numpy as np
from sklearn.datasets import fetch_openml
from PIL import Image
import pytesseract
import os

def load_mnist_data(sample_size=10000):
    print("Loading MNIST dataset (subset)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    y = y.astype(np.uint8)
    X = X / 255.0
    return X[:sample_size], y[:sample_size]

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        return img_array.flatten()
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def preprocess_text_or_doc(file_path):
    try:
        if os.path.getsize(file_path) > 500 * 1024 * 1024:  # 500 MB in bytes
            print("File size exceeds 500 MB limit!")
            return None

        text = ""
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                text = f.read().strip()
        elif file_path.endswith('.pdf'):
            import PyPDF2
            try:
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    text = pdf.pages[0].extract_text().strip()
            except Exception as e:
                print(f"PyPDF2 failed: {e}. Attempting OCR...")
                text = pytesseract.image_to_string(Image.open(file_path)).strip()
        elif file_path.endswith('.docx'):
            from docx import Document
            doc = Document(file_path)
            text = ' '.join([para.text for para in doc.paragraphs]).strip()
        
        if not text:
            print("No text found in the file!")
            return None
        
        # Extract first digit
        for char in text:
            if char.isdigit():
                return int(char)
        print("No digits found in the text!")
        return None
    except Exception as e:
        print(f"Error processing text/document: {e}")
        return None