# file_input_handler.py
from data_processing import preprocess_image, preprocess_text_or_doc
import os

def process_input_file(file_path):
    file_path = file_path.strip('"').strip("'")  # Remove quotes
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return preprocess_image(file_path)
    elif file_path.lower().endswith(('.txt', '.pdf', '.docx')):
        return preprocess_text_or_doc(file_path)
    else:
        print("Unsupported file format! Supported: .png, .jpg, .jpeg, .txt, .pdf, .docx")
        return None