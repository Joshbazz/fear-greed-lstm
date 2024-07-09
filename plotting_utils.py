import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.utils import plot_model
from IPython.display import Image, display

def plot_loss_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_plot("loss_training_history")

def plot_mae_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Model MAE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    save_plot("MAE_training_history")

def plot_predicted_actual(actual, predicted):
    df = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Actual', y='Predicted', data=df)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], color='red', linestyle='--')
    plt.title('Predicted vs. Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    save_plot("Predicted_vs_Actual")

def plot_residuals(actual, predicted):
    residuals = [actual - predicted for actual, predicted in zip(actual, predicted)]
    df = pd.DataFrame({'Actual': actual, 'Predicted': predicted, 'Residuals': residuals})
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Predicted', y='Residuals', data=df)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residuals Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    save_plot("Residuals")

def save_and_visualize_model(model, img_dir=None):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if img_dir is None:
        img_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, f"model_{timestamp}.png")
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
    print(f"Model visualization saved and displayed from {img_path}")
    # save_plot("Model_Arc")

def plot_PCA(X_scaled):
    # Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions
    X_pca = pca.fit_transform(X_scaled)

    # Plot PCA results
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.title('PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    # plt.show()

    save_plot("PCA_graph")


def save_plot(plot_name):
    current_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig(f'{plot_name}_{current_timestamp}.png')
    plt.close()
    print(f"{plot_name} plot saved as '{plot_name}_{current_timestamp}.png'")
