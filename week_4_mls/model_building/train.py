
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi(token=os.getenv("HF_TOKEN"))

Xtrain_path = "hf://datasets/NishaGok/sales-forcasting-prediction-26042026/Xtrain.csv"
Xtest_path = "hf://datasets/NishaGok/sales-forcasting-prediction-26042026/Xtest.csv"
ytrain_path = "hf://datasets/NishaGok/sales-forcasting-prediction-26042026/ytrain.csv"
ytest_path = "hf://datasets/NishaGok/sales-forcasting-prediction-26042026/ytest.csv"
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

numeric_features = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year"
]

categorical_features = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

rf_model = RandomForestRegressor(random_state=42)
param_grid = {
    "randomforestregressor__n_estimators": [100, 200],
    "randomforestregressor__max_depth": [10, 20, None],
    "randomforestregressor__min_samples_split": [2, 5],
    "randomforestregressor__min_samples_leaf": [1, 2]
}

model_pipeline = make_pipeline(preprocessor, rf_model)

grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=3,
    scoring="r2",
    verbose=1,
    n_jobs=-1
)

grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_

train_preds = best_model.predict(Xtrain)
test_preds = best_model.predict(Xtest)

print("Train R2:", r2_score(ytrain, train_preds))
print("Test R2:", r2_score(ytest, test_preds))
print("Test MAE:", mean_absolute_error(ytest, test_preds))
print("Test RMSE:", np.sqrt(mean_squared_error(ytest, test_preds)))
model_path = "best_sales_forecast_model_v1.joblib"
joblib.dump(best_model, model_path)

repo_id = "NishaGok/sales-forcasting-model-26042026"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print("Creating new model repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type,
)
