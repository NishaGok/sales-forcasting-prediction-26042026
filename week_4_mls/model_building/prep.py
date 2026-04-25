
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/NishaGok/sales-forcasting-prediction-26042026/SuperKart.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully")

# Standardize columns
# safer for pipeline consistency

df.columns = df.columns.str.replace(" ", "_")

# Drop identifiers not useful for prediction

df.drop(columns=["Product_Id", "Store_Id"], inplace=True)

TARGET_COL = "Product_Store_Sales_Total"

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="NishaGok/sales-forcasting-prediction-26042026",
        repo_type="dataset",
    )
