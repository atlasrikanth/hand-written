# visualization.py
import matplotlib.pyplot as plt

def plot_sample_digits(X, y, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 3))
    for i, ax in enumerate(axes):
        ax.imshow(X[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"Label: {y[i]}")
        ax.axis('off')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def plot_custom_input(input_data, predicted_label):
    plt.imshow(input_data.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted Digit: {predicted_label}")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)
    plt.close()