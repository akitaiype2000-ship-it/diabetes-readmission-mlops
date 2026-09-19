import pandas as pd

train = pd.read_csv("artifacts/data_ingestion/train.csv")
test = pd.read_csv("artifacts/data_ingestion/test.csv")

print(train.shape)
print(test.shape)