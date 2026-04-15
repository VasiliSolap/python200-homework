import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


# --- Task 1: Load and Explore ---
df = pd.read_csv('resources/student_performance_math.csv', sep=";")
print((f"Dataset shape: {df.shape}"))
print(df.head(5))
print(df.dtypes)

plt.figure(figsize=(10,6))
plt.hist(df["G3"], bins=21, edgecolor = "black", color = "skyblue")
plt.title("Distribution of Final Math Grades")
plt.xlabel("G3 Grade (0-20)")
plt.ylabel("Number Of Students")
plt.savefig("outputs/g3_distribution.png")
plt.show()

# --- Task 2: Preprocess the Data ---
print(f"Shape before filtering: {df.shape}")
df_clean = df[df["G3"] > 0].copy()
print(f"Shape after filtering:{df_clean.shape}")

binary_cols = ["schoolsup", "internet", "higher", "activities"]
for col in binary_cols:
    df_clean[col] = df_clean[col].map({"yes":1, "no":0})

df_clean["sex"] = df_clean["sex"].map({"F":0, "M":1})
print(df_clean.head(5))
corr_orig = df["absences"].corr(df["G3"])
corr_filtered = df_clean["absences"].corr(df_clean["G3"])

print(f"Correlation (Absences vs G3) - Original: {corr_orig:.4f}")
print(f"Correlation (Absences vs G3) - Filtered: {corr_filtered:.4f}")

# In the original data, some students with many absences had G3=0 (exam missed), 
# while others with ZERO absences also had G3=0 (missed exam for other reasons). 
# This created a "cloud" of zeros that masked the actual trend where more 
# absences usually lead to slightly lower grades

# --- Task 3: Exploratory Data Analysis ---
numeric_features = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", 
    "absences", "freetime", "goout", "Walc", "schoolsup", 
    "internet", "higher", "activities", "sex"
]

correlations = df_clean[numeric_features + ["G3"]].corr()["G3"].sort_values()
print("Correlations with G3 (Sorted):")
print(correlations)

#Impact of Past Failures on Final Grade
plt.figure(figsize=(8,5))
sns.boxplot(x = "failures", y="G3", data= df_clean, hue="failures",  palette="Set2")
plt.title("Impact of Past Failures on Final Grade")
plt.xlabel("Number of past class failures")
plt.ylabel("Final Grade (G3)")
plt.savefig('outputs/failures_vs_g3.png')
plt.show()
#As failures increase, both the median and the overall range of G3 grades drop significantly.

#Mother's Education vs Final Grade
plt.figure(figsize=(8, 5))
sns.boxplot(x='Medu', y='G3', data=df_clean, hue="failures", palette='Set2')
plt.title("Mother's Education vs Final Grade")
plt.xlabel("Mother's Education Level (0-4)")
plt.ylabel("Final Grade (G3)")
plt.savefig('outputs/medu_vs_g3.png')
plt.show()
#Students whose mothers have higher education (level 4) tend to achieve higher median grades.

# --- Task 4: Baseline Model ---
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = df_clean["failures"].values.reshape(-1,1)
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X , y, test_size=0.2, random_state=42
) 

model_base = LinearRegression()
model_base.fit(X_train, y_train)
y_pred = model_base.predict(X_test)
slope = model_base.coef_[0]
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
R2 = r2_score(y_test, y_pred)

print(f"Baseline Slope: {slope:.4f}")
print(f"Baseline RMSE: {RMSE:.4f}")
print(f"Baseline R2: {R2:.4f}")

# On a 0-20 grade scale, the slope of -1.43 means that each past failure 
# reduces the predicted final grade by nearly 1.5 points. 
# The RMSE of 2.96 indicates that our average prediction error is about 3 points, 
# which is significant. 
# An R2 of 0.089 is quite low, confirming that while past failures are a 
# statistical predictor, they explain less than 9% of the variation in grades.

# --- Task 5: Build the Full Model ---

feature_cols = [
    "failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
    "internet", "sex", "freetime", "activities", "traveltime"
]

X_full = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_full , y, test_size=0.2, random_state=42
) 

model_full = LinearRegression()
model_full.fit(X_train, y_train)

train_r2 = model_full.score(X_train, y_train)
test_r2 = model_full.score(X_test, y_test)
y_pred_f = model_full.predict(X_test)
rmse_full = np.sqrt(mean_squared_error(y_test, y_pred_f))

print(f"Full Model - Train R2: {train_r2:.4f}")
print(f"Full Model - Test R2:  {test_r2:.4f}")
print(f"Full Model - RMSE:     {rmse_full:.4f}")

print("\nFeature Coefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")

#   KEEP: 'failures', 'higher', 'studytime', 'Medu/Fedu', and 'internet'. 
# These show strong coefficients and have a clear logical link to academic success.
#   DROP: 'activities' and 'freetime'. Their coefficients are near zero (-0.009 and -0.042),
#meaning they don't help the model. Removing them simplifies data collection.

# --- Task 6: Evaluate and Summarize ---
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_f, y_test, alpha=0.6, color='royalblue', label='Predictions')

line_coords = [y_test.min(), y_test.max()]
plt.plot(line_coords, line_coords, color='red', linestyle='--', lw=2, label='Perfect Prediction')

plt.xlabel('Predicted Grades (y_hat)')
plt.ylabel('True Grades (y)')
plt.title('Predicted vs Actual (Full Model)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/predicted_vs_actual.png')
plt.show()

# 1. Dataset: 357 students (filtered), test set size is 72.
# 2. Performance: R2 0.15, RMSE 2.86. On a 0-20 scale, the model 
#    typically misses the true grade by about 3 points.
# 3. Top Drivers: 'internet' (+0.83) and 'higher' (+0.61) are the strongest 
#    positive factors. 'schoolsup' (-2.06) and 'failures' (-1.15) are the 
#    strongest negative predictors.
# 4. Surprise: 'schoolsup' has a large negative impact, likely due to 
#    "reverse causality"—it marks students who were already struggling.


# --- Neglected Feature: The Power of G1 ---
feature_cols_g1 = feature_cols + ["G1"]

X_g1 = df_clean[feature_cols_g1].values
X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)
test_r2_g1 = model_g1.score(X_test_g1, y_test_g1)

print(f"\nModel with G1 - Test R2: {test_r2_g1:.4f}")

# 1. Causality vs Correlation:
# G1 does not CAUSE G3. It is just an earlier measurement of the same student 
# ability, similar to how a thermometer measures but doesn't cause a fever.

# 2. Early vs Late Intervention:
# The G1 model is accurate but "late" (mid-semester). Educators want "early" 
# intervention. By the time G1 exists, a student might already be failing.

# 3. The Trade-off:
# Educators should use the Task 5 model for day-one predictions. Even with a 
# lower R2 (0.15), it uses "upstream" data (habits, home) to help students 
# before their first poor grade actually happens.