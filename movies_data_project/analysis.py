import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("clean_movies_data.csv")
df = df[['budget', 'runtime', 'vote_average', 'revenue']]

X = df[['budget', 'runtime', 'vote_average']]
y = df['revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "DecisionTree": DecisionTreeRegressor()
}





results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {"MSE": mse, "R2": r2}
    plt.figure()
    sns.scatterplot(x=y_test, y=preds)
    plt.xlabel("Actual Revenue")
    plt.ylabel("Predicted Revenue")
    plt.title(f"{name}: Predicted vs Actual")
    plt.savefig(f"{name}_scatter.png")

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")

df.hist(bins=30, figsize=(10, 8))
plt.tight_layout()
plt.savefig("histograms.png")

sns.boxplot(data=df)
plt.title("Boxplot for Numerical Features")
plt.savefig("boxplot.png")

print("Model Evaluation Results:")
for name, metrics in results.items():
    print(f"{name} - MSE: {metrics['MSE']:.2f}, R2: {metrics['R2']:.2f}")