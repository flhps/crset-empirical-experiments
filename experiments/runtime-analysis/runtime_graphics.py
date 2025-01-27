import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# Get the directory of the input file
input_file = "benchStCas-benchLookup-1731264729.521808.csv"
output_dir = os.path.dirname(os.path.abspath(input_file))

# Create timestamp for the filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Read the CSV file with proper column names
column_names = ["r", "s", "rhat", "p", "k", "duration", "bitsize", "lookup1k", "tries"]

df = pd.read_csv(input_file, names=column_names, sep=";", skiprows=1)

# Convert bitsize from bits to bytes
df["bitsize"] = df["bitsize"] / 8

# Create figure and axis objects with a single subplot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Set x-axis to log scale
ax1.set_xscale("log")
ax1.set_yscale("log")  # Set first y-axis to log scale

# Plot duration on the first y-axis
color = "tab:blue"
ax1.set_xlabel("Credential Capacity")
ax1.set_ylabel("Duration in Seconds", color=color)
line1 = ax1.plot(
    df["rhat"],
    df["duration"],
    color=color,
    marker="o",
    label="Duration in Seconds",
    linewidth=2,
)
ax1.tick_params(axis="y", labelcolor=color)

# Force y-axis limits for duration
duration_min = df["duration"].min()
duration_max = df["duration"].max()
ax1.set_ylim(duration_min * 0.9, duration_max * 1.1)  # Smaller scale factors

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
ax2.set_yscale("log")  # Set second y-axis to log scale
color = "tab:red"
ax2.set_ylabel("Total Size in Bytes", color=color)

# Plot bitsize
line2 = ax2.plot(
    df["rhat"],
    df["bitsize"],
    color=color,
    marker="s",
    label="Total Size in Bytes",
    linewidth=2,
)
ax2.tick_params(axis="y", labelcolor=color)

# Force y-axis limits for bitsize
bitsize_min = df["bitsize"].min()
bitsize_max = df["bitsize"].max()
ax2.set_ylim(bitsize_min * 0.8, bitsize_max * 1.2)  # Smaller scale factors

# Add legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")

# Add title
plt.title("Duration and Total Size vs Certificate Capacity")

# Add grid with minor gridlines for log scale
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.grid(True, which="minor", ls=":", alpha=0.1)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot with timestamp in filename
output_file = os.path.join(output_dir, f"duration_bitsize_plot_{timestamp}.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"\nPlot saved as: {output_file}")

# Close the plot to free memory
plt.close()

