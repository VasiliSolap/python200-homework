import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


#Load and Explore
df = pd.read_csv('resources/spambase.data', header=None)
#print(df.head(5))

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

features_to_plot = [48, 51, 56]
names = ['word_freq_free', 'char_freq_!', 'capital_run_length_total']

for idx, name in zip(features_to_plot, names):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=y, y=df[idx])
    plt.title(f'Distribution of {name} (0=Ham, 1=Spam)')
    plt.savefig(f'outputs/boxplot_{name}.png')
    plt.close()

#Prepare Your Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Number of components to explain 90% of the variance: {n_components}")

X_train_pca = pca.transform(X_train_scaled)[:, :n_components]
X_test_pca = pca.transform(X_test_scaled)[:, :n_components]

plt.plot(cumulative_variance)
plt.axhline(y=0.9, color='r', linestyle='--')
plt.savefig('outputs/pca_variance.png')
plt.close()

#A Classifier Comparison
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
print(f"KNN Unscaled Accuracy: {knn_unscaled.score(X_test, y_test):.4f}")

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
print(f"KNN Scaled Accuracy: {knn_scaled.score(X_test_scaled, y_test):.4f}")

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
print(f"KNN PCA Accuracy: {knn_pca.score(X_test_pca, y_test):.4f}")

depths = [3, 5, 10, None]
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    print(f"Tree Depth {d}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {rf.score(X_test, y_test):.4f}")

lr_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_scaled.fit(X_train_scaled, y_train)
print(f"LogReg Scaled Accuracy: {lr_scaled.score(X_test_scaled, y_test):.4f}")

lr_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_pca.fit(X_train_pca, y_train)
print(f"LogReg PCA Accuracy: {lr_pca.score(X_test_pca, y_test):.4f}")




print("Generating final Confusion Matrix for Random Forest...")
disp = ConfusionMatrixDisplay.from_estimator(
    rf, X_test, y_test, 
    display_labels=["Ham", "Spam"], 
    cmap='Greens'
)
plt.title("Confusion Matrix: Best Performing Model (RF)")
plt.savefig('outputs/best_model_confusion_matrix.png')
plt.show()

# --- TASK 3 SUMMARY & MODEL EVALUATION ---
# 1. BEST MODEL: Random Forest is the clear winner (95.55% accuracy). 
#    Its ensemble approach handles the 57 features better than any single model.

# 2. PCA vs. FULL DATA: 
#    - For KNN: PCA slightly improved results (89.69% vs 89.36%). 
#      This matches the hypothesis: PCA removed noise that confused the distance metric.
#    - For LogReg: Scaled data performed better than PCA (91.86% vs 91.10%). 
#      Discarding 10% of variance lost subtle signals LogReg needed for precision.

# 3. OVERFITTING: Decision Tree peaked at Depth 10 (92.51%). 
#    At Depth=None, training accuracy hit 99.95% while test accuracy dropped, 
#    proving that unlimited depth leads to "memorizing" noise rather than learning rules.

# 4. METRIC POSITION: Accuracy is the wrong metric for spam filters. 
#    POSITION: We must prioritize minimizing FALSE POSITIVES (Precision).
#    DEFENSE: A False Positive (blocking a real email) is a critical failure. 
#    A False Negative (spam in inbox) is just a minor annoyance. 
#    The Random Forest is best because it provides the safest balance of high precision.

# 5. CONFUSION MATRIX: The best model (RF) shows fewer False Positives 
#    than False Negatives, confirming it is safe for production use.