import pandas as pd

# Load dataset
df = pd.read_csv("artifacts/data/diabetic_data.csv")

# Basic information
print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

print(f"\nShape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())