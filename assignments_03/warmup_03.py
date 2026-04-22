import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---

#Preprocessing Q1
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

print(f"X_train Shape: {X_train.shape}")
print(f"X_test Shape: {X_test.shape}")
print(f"y_train Shape: {y_train.shape}")
print(f"y_test Shape: {y_test.shape}")

#Preprocessing Q2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Means of columns in X_train_scaled:")
print(X_train_scaled.mean(axis=0))

# Fit the scaler on the training data only to prevent data leakage, 
# ensuring the test set parameters remain completely unknown to the model until evaluation.


# --- KNN ---

#KNN Q1
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

#KNN Q2
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
preds = knn.predict(X_test_scaled)

print("Accuracy (scaled data):", accuracy_score(y_test, preds))

# Scaling doesn't change performance here because Iris features are already in 
# the same units (cm), have similar ranges, and provide clear natural separation.

#KNN Q3
knn = KNeighborsClassifier(n_neighbors=5)

cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
print("Scores for each fold:", cv_scores)
print(f"Mean Accuracy: {cv_scores.mean():.3f}")
print(f"Standard Deviation: {cv_scores.std():.3f}")

# This result is more trustworthy than a single train/test split because it 
# provides a robust average performance across multiple data subsets, reducing 
# the impact of a single "lucky" or "unlucky" random data partition.

#KNN Q4
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}: mean accuracy = {scores.mean():.3f}")

# I would choose the k that yields the highest mean CV score, as it represents 
# the best balance between bias and variance for this specific dataset. 
# On the Iris dataset, multiple k values often show similar high performance 
# due to its clean nature.

# --- Classifier Evaluation ---

#Classifier Evaluation Q1
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("KNN Confusion Matrix (Iris)")
plt.savefig("outputs/knn_confusion_matrix.png")
plt.show()

# For the Iris dataset, if errors occur, the model typically confuses Versicolor and Virginica 
# because their features overlap more than those of the Setosa species.

# --- The sklearn API: Decision Trees ---

#Decision Trees Q1
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

print(f"Decision Trees Accyracy:{accuracy_score(y_test, dt_preds):.3f}")
print("\nClassification Report:\n", classification_report(y_test, dt_preds))

# Comparison to KNN:
# On the Iris dataset, the Decision Tree usually achieves a high accuracy 
# (often around 0.96-1.0), which is very competitive with KNN. However, KNN 
# with k=5 sometimes performs slightly better on very small, clean datasets.

# Impact of scaling:
# Scaling (StandardScaler) would not affect the Decision Tree's results. 
# Since the algorithm splits data based on value thresholds rather than 
# distance calculations, it is invariant to monotonic transformations.

# --- Logistic Regression and Regularization ---

#Logistic Regression Q1
c_values = [0.01, 1.0, 100]

for c in c_values:
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    coef_magnitude = np.abs(model.coef_).sum()
    
    print(f"C = {c:4}: Total coefficient magnitude = {coef_magnitude:.4f}")

# Comment on the results:
# As C increases, the total magnitude of the coefficients also increases. 
# This shows that regularization (small C) penalizes large weights, forcing 
# them closer to zero to prevent overfitting and make the model "simpler". 
# A large C allows the model to assign more importance (larger weights) 
# to features to fit the training data more closely.

# --- PCA ---
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images = digits.images  # same data shaped as 8x8 images for plotting

#PCA Q1
print(f"Shape of X_digits (flattened): {X_digits.shape}")
print(f"Shape of images (8x8): {images.shape}")

fig, axes = plt.subplots(1,10, figsize = (15,3))
for i in range(10):
    index = (y_digits == i).argmax()
    axes[i].imshow(images[index], cmap = "gray_r")
    axes[i].set_title(f"Label:{i}")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig('outputs/sample_digits.png')
plt.show()

#PCA Q2
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

plt.figure(figsize=(10,8))

scatter = plt.scatter(scores[:,0], scores[:,1], c=y_digits, cmap="tab10", s=10)
plt.colorbar(scatter, label="Digit")
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.title('PCA 2D Projection of Digits Dataset')
plt.savefig('outputs/pca_2d_projection.png')
plt.show()

# Yes, same-digit images definitely tend to cluster together! 
# Even in just 2 dimensions, we can see distinct groups (like the clusters for '0' or '4'). 
# Although some digits overlap in the center, PCA has managed to separate most 
# of the 64-dimensional information into a readable 2D map.

#PCA Q3
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', 
         linestyle='--', markersize=3)
plt.axhline(y=0.8, color='r', linestyle=':', label='80% Threshold')

plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig('outputs/pca_variance_explained.png')
plt.show()

# To explain about 80% of the variance, we need approximately 13 components. 
# It's amazing that we can drop from 64 features down to around 13 and still 
# keep 80% of the original information! This makes the model much lighter 
# without losing the "essence" of the handwritten digits.

#PCA Q4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_components_list = [2,5,15,40]
n_digits = 5

fig, axes = plt.subplots(len(n_components_list) + 1, n_digits, figsize=(10, 10))


for j in range(n_digits):
    axes[0, j].imshow(images[j], cmap='gray_r')
    axes[0, j].set_title(f"Original {y_digits[j]}")
    axes[0, j].axis('off')


for row_idx, n in enumerate(n_components_list, start=1):
    for col_idx in range(n_digits):
        reconstructed_img = reconstruct_digit(col_idx, scores, pca, n)
        axes[row_idx, col_idx].imshow(reconstructed_img, cmap='gray_r')
        axes[row_idx, col_idx].set_title(f"n={n}")
        axes[row_idx, col_idx].axis('off')

plt.tight_layout()
plt.savefig('outputs/pca_reconstructions.png')
plt.show()


# The digits become clearly recognizable at n=15. At n=2 and n=5, 
# they look like blurry ghosts or ink stains. 
# We can see that 15 components capture about 80% of the variance, 
# and here we can see that 80%. By n=40, it's almost identical to the original.