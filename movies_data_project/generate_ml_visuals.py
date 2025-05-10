import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# لو بتستخدم clean_movie_data()
from clean_data import clean_movie_data

# Load data
df = clean_movie_data()

# اختار فقط الأعمدة الرقمية المفيدة
features = ['budget', 'runtime', 'vote_average']
target = 'revenue'

df_model = df[features + [target]].dropna()

# Split
X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Heatmap ----------------
plt.figure(figsize=(8, 6))
sns.heatmap(df_model.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.close()

# ---------------- Linear Regression ----------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Linear Regression: Actual vs Predicted")
plt.tight_layout()
plt.savefig("LinearRegression_scatter.png")
plt.close()

# ---------------- Decision Tree ----------------
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_dt, alpha=0.5, color='green')
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Decision Tree: Actual vs Predicted")
plt.tight_layout()
plt.savefig("DecisionTree_scatter.png")
plt.close()

print("✅ Images generated successfully.")


