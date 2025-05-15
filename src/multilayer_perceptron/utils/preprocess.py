import pandas as pd
from sklite.preprocessing import LabelEncoder, StandardScaler
from multilayer_perceptron.config import TARGET_FEATURE, DATASET_COLUMNS, DATASET_NAME


# This function is used to preprocess the data for the certain dataset.
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    print(f"Preprocessing {DATASET_NAME} dataset...")
    # 1. Drop the ID column
    data = data.drop(columns=["ID"])

    # 2. Encode the target feature
    label_encoder = LabelEncoder(columns=[TARGET_FEATURE])
    label_encoder.fit(data)
    data = label_encoder.transform(data)

    # 3. Split the data into features and target
    X = data.drop(columns=[TARGET_FEATURE])
    y = data[TARGET_FEATURE]

    # 4. Normalize the features
    scaler = StandardScaler(columns=X.columns.tolist())
    scaler.fit(X)
    X = scaler.transform(X)

    # 5. Convert the data back to a DataFrame
    data = pd.DataFrame(X, columns=DATASET_COLUMNS[2:])
    data[TARGET_FEATURE] = y
    data = data.astype({TARGET_FEATURE: "int"})
    return data
