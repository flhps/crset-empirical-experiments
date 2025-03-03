import sys
import os
sys.path.append(os.path.abspath('./../..'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.utils.cascade_property_vectorizer import get_vector_from_string

# Create images directory if it doesn't exist
IMAGE_DIR = './images'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def save_figure(plt, name_prefix):
    """Helper function to save figures with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'padded_{name_prefix}_{timestamp}.png'
    filepath = os.path.join(IMAGE_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {filepath}")

# Set parameters
file_path = "./../../data/with_padding/single/single-data-1736344150207314900.csv"
chunk_size = 1000
test_size = 0.2
random_state = 42

# Initialize lists for collecting data
X_all = []
y_all = []

# Read and process data in chunks
print("Reading and processing data...")
for chunk in pd.read_csv(file_path, chunksize=chunk_size, delimiter=";"):
    X_chunk = np.array([get_vector_from_string(row)
                        for row in chunk['concatenated_bitstrings']])
    y_chunk = chunk['num_included'].values
    X_all.append(X_chunk)
    y_all.append(y_chunk)

# Combine chunks into one array
X = np.vstack(X_all)
y = np.concatenate(y_all)

feature_names = [
    'Total Size', 'Num Filters', 'Filter1 Size', 
    'Filter2 Size', 'Filter3 Size', 'Filter1 Set Bits',
    'Filter2 Set Bits', 'Filter3 Set Bits', 'F1 F2 Ratio'
]
df = pd.DataFrame(X, columns=feature_names)

# Create scatter plots of features vs num_included
plt.figure(figsize=(20, 15))
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Features vs num_included (Padded Certificates)', size=16, y=1.02)

axes = axes.ravel()

for idx, feature in enumerate(feature_names):
    sns.scatterplot(
        data=df,
        x=y,
        y=feature,
        ax=axes[idx],
        alpha=0.5,
        size=0.25
    )
    axes[idx].set_title(f'num_included vs {feature}')
    axes[idx].set_xlabel('num_included')
    axes[idx].set_ylabel(feature)

fig.delaxes(axes[-1])
plt.tight_layout()
save_figure(plt, 'features_vs_num_included')

# Scale features and split data
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=random_state
)

# Train model and make predictions
print("Training model...")
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate and print metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance Metrics:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.axhline(y=np.mean(y_pred), color='r', linestyle='--', 
            label=f'Mean prediction: {np.mean(y_pred):.2f}')

y_min, y_max = np.min(y_pred), np.max(y_pred)
plt.ylim(y_min - (y_max - y_min)*0.05, y_max + (y_max - y_min)*0.05)

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Padded Certificates)')
plt.legend()
plt.tight_layout()
save_figure(plt, 'actual_vs_predicted')

# Print feature coefficients
coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
})
coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Coefficients (sorted by absolute value):")
for _, row in coefficients.iterrows():
    print(f"{row['Feature']}: {row['Coefficient']:.4f}")