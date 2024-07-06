import os
from datetime import datetime
from tensorflow.keras.utils import plot_model
from IPython.display import Image, display
import matplotlib.pyplot as plt
import datetime

def plot_loss_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    current_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig(f'loss_training_history_{current_timestamp}.png')
    plt.close()  # Close the plot to free up memory

    print(f"Training history plot saved as 'loss_training_history_{current_timestamp}.png'")


def plot_mse_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Model MSE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    current_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig(f'MAE_training_history_{current_timestamp}.png')
    plt.close()  # Close the plot to free up memory

    print(f"Training history plot saved as 'MAE_training_history_{current_timestamp}.png'")


def save_and_visualize_model(model, img_dir='Fear_Greed/visualization'):
    """
    Save the loaded model and visualize it as a PNG image.

    Parameters:
    - model: The Keras model to be visualized.
    - img_dir: The directory where the visualization image should be saved.
    """

    # Generate a timestamp for the model file
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Use the directory of this script if img_dir is not provided
    if img_dir is None:
        img_dir = os.path.dirname(os.path.abspath(__file__))

    # Ensure the directory exists
    os.makedirs(img_dir, exist_ok=True)

    # Define the image path
    img_path = os.path.join(img_dir, f"model_{timestamp}.png")

    # Plot the model and save it as a PNG image
    plot_model(
        model,
        to_file=img_path,
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True
    )

    # Display the image
    # img = Image(filename=img_path)
    # display(img)
    print(f"Model visualization saved and displayed from {img_path}")