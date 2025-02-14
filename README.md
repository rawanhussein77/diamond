import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso


data = pd.read_csv("diamonds.csv", index_col=0)

data["volume"] = data["x"] * data["y"] * data["z"]
data["depth_ratio"] = data["depth"] / data["table"]
data = data.drop(columns=["x", "y", "z"])


categorical_columns = ["cut", "color", "clarity"]
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)


X = data.drop(columns=["price"])
y = data["price"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


X_train_svm, _, y_train_svm, _ = train_test_split(X_train, y_train, test_size=0.7, random_state=42)


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.8, max_iter=10000),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Bagging Regressor": BaggingRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "SVM": SVR(kernel="linear")  
}


results = {}
for name, model in models.items():
    if name == "SVM":
        mse, r2 = train_and_evaluate(model, X_train_svm, y_train_svm, X_test, y_test)
    else:
        mse, r2 = train_and_evaluate(model, X_train, y_train, X_test, y_test)
    results[name] = {"MSE": mse, "R2 Score": r2}


results_df = pd.DataFrame(results).T
print(results_df)
