# main.py
import numpy as np
from data_processing import load_mnist_data
from model import train_model, predict_digit
from visualization import plot_sample_digits, plot_custom_input
from file_input_handler import process_input_file

def main():
    X, y = load_mnist_data(sample_size=10000)
    print("Visualizing sample MNIST digits...")
    plot_sample_digits(X, y)

    model = train_model(X, y, use_knn=True)

    while True:
        file_path = input("\nEnter the path to your file (e.g., C:\\path\\to\\file.png, < 500 MB) or 'exit' to quit: ")
        if file_path.lower() == 'exit':
            break
        
        input_data = process_input_file(file_path)
        if input_data is None:
            continue
        
        if isinstance(input_data, np.ndarray):  # Image input
            predicted_digit = predict_digit(model, input_data)
            print(f"Predicted digit from image: {predicted_digit}")
            plot_custom_input(input_data, predicted_digit)
        else:  # Text-based input
            print(f"Extracted digit from file: {input_data}")
            print("No prediction needed for direct digit input.")

if __name__ == "__main__":
    main()