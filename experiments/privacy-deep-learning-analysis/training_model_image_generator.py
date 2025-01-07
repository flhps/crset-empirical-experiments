import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

# Set style for scientific publication
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12  # Base font size increased
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 11  # Increased tick label size
mpl.rcParams['ytick.labelsize'] = 11  # Increased tick label size
mpl.rcParams['figure.figsize'] = (10, 6)

# Load the training histories
with open('./training_history_padded.json', 'r') as f:
    history_padded = json.load(f)

with open('./training_history_unpadded.json', 'r') as f:
    history_unpadded = json.load(f)

# Create figure and axis
fig, ax = plt.subplots()

# Plot training MAE
epochs = range(1, len(history_padded['history']['mae']) + 1)

# Plot padded dataset (blues)
ax.plot(epochs, history_padded['history']['mae'], 
        color='#1f77b4', linestyle='--', linewidth=2, 
        label='Training MAE (Padded)', alpha=0.8)
ax.plot(epochs, history_padded['history']['val_mae'], 
        color='#7cc7ff', linestyle='-', linewidth=2, 
        label='Validation MAE (Padded)', alpha=0.8)

# Plot unpadded dataset (reds)
ax.plot(epochs, history_unpadded['mae'], 
        color='#d62728', linestyle='--', linewidth=2, 
        label='Training MAE (Unpadded)', alpha=0.8)
ax.plot(epochs, history_unpadded['val_mae'], 
        color='#ff9896', linestyle='-', linewidth=2, 
        label='Validation MAE (Unpadded)', alpha=0.8)

# Customize the plot
ax.set_xlabel('Epoch')
ax.set_ylabel('Mean Absolute Error')
ax.set_title('Model Training and Validation MAE Over Time')
ax.grid(True, linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add legend inside the plot at top right
ax.legend(frameon=True, facecolor='white', framealpha=1, 
          edgecolor='none', loc='upper right', fontsize=12)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Generate timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save the plot with timestamp
plt.savefig(f'training_history_comparison_{timestamp}.png', 
            dpi=300, bbox_inches='tight')
plt.close()