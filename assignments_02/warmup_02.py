# --- scikit-learn API ---

#Q1
import numpy as np
from sklearn.linear_model import LinearRegression

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)

new_salary = np.array([4, 8]).reshape(-1, 1)
predictions = model.predict(new_salary)

print(f"Slope: {model.coef_[0]: .2f}")
print(f"Intersept: {model.intercept_: .2f}")
print(f"Prediction for 4 years: {predictions[0]: .2f}")
print(f"Prediction for 8 years: {predictions[1]: .2f}")

#Q2
x = np.array([10, 20, 30, 40, 50])
print(f"Original shape: {x.shape}")

x_2D = x.reshape(-1,1)
print(f"Reshape to 2D: {x_2D.shape}")
# Scikit-learn requires the feature array X to be 2D because it expects 
# data to be structured as a table or matrix

#Q3
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Points per cluster:", np.bincount(labels))
plt.figure(figsize=(8, 6))
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', s=200, label='Centers')

plt.title("K-Means Clustering: 3 Groups Found") 
plt.xlabel("Feature 1") # [cite: 372, 644]
plt.ylabel("Feature 2") # [cite: 372, 644]
plt.legend()

plt.savefig('outputs/kmeans_clusters.png')
plt.show()


# --- Linear Regression ---

#Q1
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)
plt.figure(figsize=(10,6))
plt.scatter(age, cost, c = smoker, cmap="coolwarm", alpha=0.7)
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost($)")
plt.savefig('outputs/cost_vs_age.png')
plt.show()

# The plot clearly shows two distinct parallel groups of data points: 
# red (smokers) and blue (non-smokers). 
# According to the graph, medical costs increase with age, 
# but smokers have significantly higher expenses compared to non-smokers.

#Q2
X = age.reshape(-1, 1)
y = cost
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", y_train.shape)
print("Y_test shape:", y_test.shape)

#Q3
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print(f"Slope: {model.coef_[0]: .2f}")
print(f"Intersept: {model.intercept_: .2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.2f}")

#The slope indicates the predicted increase in medical 
#costs for each additional year of age.

#Q4
X_full = np.column_stack([age, smoker])
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_full, y, test_size=0.2, random_state=42)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)
r2_full = model_full.score(X_test_f, y_test_f)

print(f"Full Model Test R²: {r2_full:.4f}")
print(f"Age coefficient:    {model_full.coef_[0]:.2f}")
print(f"Smoker coefficient: {model_full.coef_[1]:.2f}")

#In practical terms, the smoker coefficient represents the estimated average 
#increase in annual medical costs for being a smoker compared to a non-smoker

#Q5
plt.figure(figsize=(8,8))
plt.scatter(y_pred, y_test_f, color = "blue", alpha=0.6, label = "Predictions")
plt.title("Predicted vs Actual")

max_val = max(max(y_pred), max(y_test_f))
min_val = min(min(y_pred), min(y_test_f))

plt.plot([min_val, max_val], [min_val, max_val], color = "red", linestyle= '--', label = "Perfect Fit")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost ($)")
plt.ylabel("Actual Cost ($)")
plt.legend()
plt.grid(True, alpha=0.3)


plt.savefig('outputs/predicted_vs_actual.png')
plt.show()