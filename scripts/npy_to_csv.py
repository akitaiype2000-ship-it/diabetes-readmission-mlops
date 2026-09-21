import numpy as np
import pandas as pd
import os

# Load transformed training data
train = np.load("artifacts/data_transformation/train.npy", allow_pickle=True)

# Create column names
columns = [f"feature_{i}" for i in range(train.shape[1] - 1)]
columns.append("target")

df = pd.DataFrame(train, columns=columns)

# Add columns required by Feast
df["patient_id"] = range(1, len(df) + 1)
df["event_timestamp"] = pd.Timestamp.now()

# Create output folder
os.makedirs("feature_repo/feature_repo/data", exist_ok=True)

# Save CSV
df.to_csv(
    "feature_repo/feature_repo/data/diabetes_features.csv",
    index=False
)

print("CSV created successfully!")