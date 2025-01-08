import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from datetime import datetime
import os

# Create directory for saving images if it doesn't exist
IMAGE_DIR = './images'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def save_figure(plt, name_prefix):
    """Helper function to save figures with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{name_prefix}_{timestamp}.png'
    filepath = os.path.join(IMAGE_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {filepath}")

def load_data(filepath='./../../data/with_padding/single/single-data-1736344150207314900.csv'):
    df = pd.read_csv(filepath, delimiter=';')
    X = df['concatenated_bitstrings'].str.split(',').apply(lambda x: ''.join(x))
    X = np.array([list(x) for x in X]).astype(int)
    y = df['num_excluded'].values
    return X, y

def plot_distribution(y):
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    })
    
    plt.figure(figsize=(10, 6))
    plt.hist(y, bins=30, edgecolor='black')
    plt.title('Distribution of num_excluded Values')
    plt.xlabel('num_excluded')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.axvline(np.mean(y), color='red', linestyle='dashed', 
                linewidth=2, label=f'Mean: {np.mean(y):.2f}')
    plt.axvline(np.median(y), color='green', linestyle='dashed', 
                linewidth=2, label=f'Median: {np.median(y):.2f}')
    plt.legend()
    plt.tight_layout()
    
    save_figure(plt, 'distribution')

def create_model(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(192, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(64, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(),
        
        keras.layers.Dense(32, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(0.01)),
        
        keras.layers.Dense(1)
    ])
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    save_figure(plt, 'training_history')

def plot_model_comparison():
    # Set style for scientific publication
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (10, 6)
    })
    
    # Load histories
    with open('./training_history_padded.json', 'r') as f:
        history_padded = json.load(f)
    with open('./training_history_unpadded.json', 'r') as f:
        history_unpadded = json.load(f)
    
    fig, ax = plt.subplots()
    epochs = range(1, len(history_padded['mae']) + 1)
    
    # Plot padded dataset
    ax.plot(epochs, history_padded['mae'],
            color='#1f77b4', linestyle='--', linewidth=2,
            label='Training MAE (Padded)', alpha=0.8)
    ax.plot(epochs, history_padded['val_mae'],
            color='#7cc7ff', linestyle='-', linewidth=2,
            label='Validation MAE (Padded)', alpha=0.8)
    
    # Plot unpadded dataset
    ax.plot(epochs, history_unpadded['mae'],
            color='#d62728', linestyle='--', linewidth=2,
            label='Training MAE (Unpadded)', alpha=0.8)
    ax.plot(epochs, history_unpadded['val_mae'],
            color='#ff9896', linestyle='-', linewidth=2,
            label='Validation MAE (Unpadded)', alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Model Training and Validation MAE Over Time')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(frameon=True, facecolor='white', framealpha=1,
              edgecolor='none', loc='upper right', fontsize=12)
    
    plt.tight_layout()
    save_figure(plt, 'model_comparison')

def main():
    # Load and prepare data
    X, y = load_data()
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    
    # Plot distribution
    plot_distribution(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Create and train model
    model = create_model(X_train.shape[1])
    optimizer = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_delta=0.05,
        min_lr=0.0001,
        verbose=1
    )
    
    model.compile(
        optimizer=optimizer,
        loss='huber',
        metrics=['mae']
    )
    
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        callbacks=[reduce_lr],
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save training history
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    with open('training_history_padded.json', 'w') as f:
        json.dump(history_dict, f)
    
    # Plot model comparison
    plot_model_comparison()
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

if __name__ == "__main__":
    main()