
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


file_path = "dataset.ods" 

xls = pd.ExcelFile(file_path, engine="odf")
leagues =xls.sheet_names

print(leagues)

for league in leagues:
    df = pd.read_excel(file_path, engine="odf", sheet_name=league)
    #print(df.head())

    features = [
        "Expenditure",
        "Income",
        "Squad Value",
        "Avg. Squad Age"
        #"Goals For",
        #"Goals Against"
    ]

    target = "Points per game"

    df = df[features + [target]]

    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('€', '', regex=False)
            .str.replace('m', 'e6', regex=False)
            .str.replace('k', 'e3', regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()
    print("\n"+league)
    print("Cleaned data shape:", df.shape)

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("Model Performance:")
    print("R²:", round(r2, 3))
    print("RMSE:", round(rmse, 3))
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("Feature Importance (RF):")
    print(importance_df)



