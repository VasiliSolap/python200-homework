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
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# --- Task 1: Load and Explore ---
df = pd.read_csv('resources/spambase.data', header=None)

feature_names = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
    'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
    'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
    'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
    'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
    'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
    'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
    'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
    'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',
    'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
    'char_freq_$', 'char_freq_#', 'capital_run_length_average',
    'capital_run_length_longest', 'capital_run_length_total'
]
#shape
print(f"Dataset shape: {df.shape}")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#Class balance
print("\nClass balance:")
print(y.value_counts())

# Class balance shows how many ham vs spam emails are in the dataset.
# This is significant because if one class dominates,
# the model could achieve high accuracy by always predicting the majority class (ham),
# while completely ignoring the minority class (spam).

features_to_plot = [48, 51, 56]
names = ['word_freq_free', 'char_freq_!', 'capital_run_length_total']

for idx, name in zip(features_to_plot, names):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=y, y=df[idx])
    plt.title(f'Distribution of {name} (0=Ham, 1=Spam)')
    plt.savefig(f'outputs/boxplot_{name}.png')
    plt.close()

# --- Task 2: Prepare Your Data ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

# --- Task 3: A Classifier Comparison ---
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
knn_preds = knn_unscaled.predict(X_test)
print(f"KNN Unscaled Accuracy: {knn_unscaled.score(X_test, y_test):.4f}")
print(classification_report(y_test,knn_preds, target_names=['Ham', 'Spam']))

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
knn_scaled_pred = knn_scaled.predict(X_test_scaled)
print(f"KNN Scaled Accuracy: {knn_scaled.score(X_test_scaled, y_test):.4f}")
print(classification_report(y_test,knn_scaled_pred, target_names=['Ham', 'Spam']))

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
knn_pca_pred = knn_pca.predict(X_test_pca)
print(f"KNN PCA Accuracy: {knn_pca.score(X_test_pca, y_test):.4f}")
print(classification_report(y_test,knn_pca_pred, target_names=['Ham', 'Spam']))

depths = [3, 5, 10, None]
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    print(f"Tree Depth {d}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")
# Depth=10 achieves the highest test accuracy (92.5%) with acceptable
# overfitting gap (4.6%). Depth=None overfits: train 99.9% vs test 91.9%.
dt_best = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_best.fit(X_train, y_train)
dt_pred = dt_best.predict(X_test)
print(classification_report(y_test, dt_pred, target_names=['Ham', 'Spam']))

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f"Random Forest Accuracy: {rf.score(X_test, y_test):.4f}")
print(classification_report(y_test, rf_pred, target_names=['Ham', 'Spam']))

lr_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_scaled.fit(X_train_scaled, y_train)
lr_pred = lr_scaled.predict(X_test_scaled)
print(f"LogReg Scaled Accuracy: {lr_scaled.score(X_test_scaled, y_test):.4f}")
print(classification_report(y_test, lr_pred, target_names=['Ham', 'Spam']))

lr_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_pca.fit(X_train_pca, y_train)
lr_pca_pred = lr_pca.predict(X_test_pca)
print(f"LogReg PCA Accuracy: {lr_pca.score(X_test_pca, y_test):.4f}")
print(classification_report(y_test, lr_pca_pred, target_names=['Ham', 'Spam']))

importances_dt = dt_best.feature_importances_
indices_dt = np.argsort(importances_dt)[::-1][:10]
print("Top 10 features (Decision Tree):")
for rank, i in enumerate(indices_dt, 1):
    print(f"{rank}. {feature_names[i]}: {importances_dt[i]:.4f}")

# Top 10 features - Random Forest + bar chart
importances_rf = rf.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1][:10]
print("\nTop 10 features (Random Forest):")
for rank, i in enumerate(indices_rf, 1):
    print(f"{rank}. {feature_names[i]}: {importances_rf[i]:.4f}")

# Bar chart для Random Forest
plt.figure(figsize=(10, 6))
plt.bar(range(10), importances_rf[indices_rf])
plt.xticks(range(10), [feature_names[i] for i in indices_rf], rotation=45, ha='right')
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig('outputs/feature_importances.png')
plt.show()

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
# 1. BEST MODEL: Random Forest (94.57%) — best accuracy overall.
#    Its ensemble of 100 trees handles 57 features better than any single model.

# 2. PCA COMPARISON:
#    KNN: Scaled (90.77%) slightly better than PCA (90.66%).
#    LogReg: Scaled (92.94%) better than PCA (91.86%).
#    PCA slightly hurt both models — dropping 10% of variance lost useful signal.

# 3. OVERFITTING: Decision Tree peaked at Depth 10 (90.88%).
#    At Depth=None, training accuracy hit 99.97% while test accuracy dropped,
#    proving that unlimited depth leads to memorizing noise rather than learning rules.

# 4. METRIC FOR SPAM FILTER: Precision is more important than accuracy.
#    A False Positive (ham marked as spam) means losing an important email — critical failure.
#    A False Negative (spam in inbox) is just a minor annoyance.
#    Random Forest has the best precision for spam (0.95) — safest choice for production.

# --- Task 4: Cross-Validation ---

classifiers = [
    ("KNN Unscaled",    knn_unscaled,  X_train),
    ("KNN Scaled",      knn_scaled,    X_train_scaled),
    ("KNN PCA",         knn_pca,       X_train_pca),
    ("Decision Tree",   dt_best,       X_train),
    ("Random Forest",   rf,            X_train),
    ("LogReg Scaled",   lr_scaled,     X_train_scaled),
    ("LogReg PCA",      lr_pca,        X_train_pca),
]

for name, model, X in classifiers:
    scores = cross_val_score(model, X, y_train, cv=5)
    print(f"{name:20s}: mean={scores.mean():.4f}, std={scores.std():.4f}")

# 1. MOST ACCURATE: Random Forest (mean=0.9543) — best overall.
# 2. MOST STABLE: LogReg PCA (std=0.0034) — lowest variance across folds.
# 3. RANKING: matches Task 3 results — Random Forest leads,
#    KNN Unscaled is worst. Cross-validation confirms our earlier conclusions.

# --- Task 5: Build your pipelines ---

rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_pipe_pred = rf_pipeline.predict(X_test)
print("\nRandom Forest Pipeline:")
print(classification_report(y_test, rf_pipe_pred, target_names=['Ham', 'Spam']))

lr_pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

lr_pipeline.fit(X_train, y_train)
lr_pipe_pred = lr_pipeline.predict(X_test)
print("\nLogReg Pipeline:")
print(classification_report(y_test, lr_pipe_pred, target_names=['Ham', 'Spam']))

# Pipeline structures are different:
# RF pipeline: classifier only — trees don't need scaling
# LR pipeline: scaler and classifier — LogReg needs scaled features

# Practical value of pipelines:
# 1. No data leakage — scaler fits on train only, automatically
# 2. Less bookkeeping — one fit() and predict() instead of many steps
# 3. Easy to deploy — hand off one object instead of scaler + model separately