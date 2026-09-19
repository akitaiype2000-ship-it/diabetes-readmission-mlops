import pandas as pd

df = pd.read_csv("artifacts/data/diabetic_data.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nTarget Distribution:")
print(df["readmitted"].value_counts())

print("\nPercentage Distribution:")
print(df["readmitted"].value_counts(normalize=True) * 100)

print("\nData Types:")
print(df.dtypes.value_counts())

print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\nUnique Values per Column:")
print(df.nunique().sort_values())