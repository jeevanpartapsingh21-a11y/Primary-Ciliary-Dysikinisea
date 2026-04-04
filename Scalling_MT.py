import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

INFILE  = r"C:\Users\Ssehra\Desktop\PCD\output\CIliary Beat Frequency\Mouse_trachea_unscaled.csv"
OUTFILE = r"C:\Users\Ssehra\Desktop\PCD\output\CIliary Beat Frequency\Scaled.MT.csv"


MOUSE_LABEL = "Group"
FEATURE_COLS = ["CBF", "SEM"]

# Load
mouse = pd.read_csv(INFILE)

print("Shape:", mouse.shape)
print("Columns:", list(mouse.columns))
print(mouse.head(3))

# Ensure numeric
for c in FEATURE_COLS:
    mouse[c] = pd.to_numeric(mouse[c], errors="coerce")

# Clean rows with missing label or features
mouse_clean = mouse.dropna(subset=FEATURE_COLS + [MOUSE_LABEL]).reset_index(drop=True)

print("\nAfter cleaning:")
print("Shape:", mouse_clean.shape)
print("Group counts:\n", mouse_clean[MOUSE_LABEL].value_counts())

mouse_clean.to_csv(OUTFILE, index=False)
print(f"\nSaved cleaned file: {OUTFILE}")

print("\nPreview:")
print(mouse_clean.head(3))