# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/JyalaHarsha-2025/MLOPS_Tourism_Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)

# Target column
target_col = 'prodtaken'

# Define categorical columns for label encoding
categorical_columns = [
    'typeofcontact', 'citytier', 'occupation', 'gender', 'productpitched',
    'maritalstatus', 'passport', 'owncar', 'designation'
]

# Handle missing values
# For numerical columns, fill with median
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_columns.remove('prodtaken')  # Remove target if included

for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())

# For categorical columns, fill with mode
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # in case you want to reuse them

# Encode target column
target_encoder = LabelEncoder()
df[target_col] = target_encoder.fit_transform(df[target_col])

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload to Hugging Face
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="JyalaHarsha-2025/MLOPS_Tourism_Prediction",
        repo_type="dataset",
    )
