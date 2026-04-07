import pandas as pd

# --- Pandas ---

# Pandas Q1
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print("First 3 rows:")
print(df.head(3))

print(f"Shape: {df.shape}")
print(f"Num Rows: {len(df)}")

print("Data types:")
print(df.dtypes)

# Pandas Q2
filtered = df[(df["passed"] == True) & (df["grade"] > 80)]
print("Filtered:")
print(filtered)

# Pandas Q3
df["grade_curved"] = df["grade"] + 5
print("Updated DataFrame:")
print(df)

# Pandas Q4
df["name_upper"] = df["name"].str.upper()
print("Name and Name Upper:")
print(df[["name", "name_upper"]])

# Pandas Q5
city_grades = df.groupby("city")["grade"].mean()
print("Groupby City by Grade:")
print (city_grades)

# Pandas Q6
df["city"] = df["city"].replace("Austin", "Houston")
print("City replace:")
print(df[["name", "city"]])

# Pandas Q7
sorted=df.sort_values("grade", ascending=False)
print("Top 3 by Grade:")
print(sorted.head(3))

# NumPy
import numpy as np

# NumPy Q1
arr = np.array([10, 20, 30, 40, 50])
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")
print(f"Ndim: {arr.ndim}")

# NumPy Q2
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print(f"Shape: {arr.shape}")
print(f"Size: {arr.size}")

# NumPy Q3
print("Top-left 2x2 block:")
print(arr[0:2, 0:2])

# NumPy Q4
zeros = np.zeros((3, 4))
ones = np.ones((2, 5))
print("Array of zeros:")
print(zeros)
print("Array of ones:")
print(ones)

# NumPy Q5
arr=np.arange(0, 50, 5)
print(f"Array: {arr}")
print(f"Shape:{arr.shape}")
print(f"Mean:{arr.mean()}")
print(f"Sum:{arr.sum()}")
print(f"Standard deviation:{arr.std()}")

# NumPy Q6
arr = np.random.normal(0, 1, 200)
print(f"Mean:{arr.mean()}")
print(f"Standard deviation:{arr.std()}")


# --- Matplotlib ---
import matplotlib.pyplot as plt

# Matplotlib Q1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y)          
plt.title("Squares")  
plt.xlabel("X")     
plt.ylabel("Y")     
plt.savefig("assignments_01/outputs/plt_line.png")
plt.close()
print("Matplotlib Q1 saved")            

# Matplotlib Q2
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.bar(subjects,scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")     
plt.ylabel("Scores") 
plt.savefig("assignments_01/outputs/plt_bar.png")
plt.close()
print("Matplotlib Q2 saved") 

# Matplotlib Q3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color="blue", label="Dataset 1")
plt.scatter(x2, y2, color="red", label="Dataset 2")
plt.title("Two Datasets")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend() 
plt.savefig("assignments_01/outputs/plt_scatter.png")
plt.close()
print("Matplotlib Q3 saved") 

# Matplotlib Q4
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(x,y)
ax1.set_title("Left")

ax2.bar(subjects, scores)
ax2.set_title("Right")

plt.tight_layout()
plt.savefig("assignments_01/outputs/subplots.png")
plt.close()
print("Matplotlib Q1 saved") 

# Descriptive Stats Q1
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Variance: {np.var(data)}")
print(f"Standard Deviation: {np.std(data)}")

# Descriptive Stats Q2
arr = np.random.normal(65, 10, 500)
plt.hist(arr, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.savefig("assignments_01/outputs/distribution.png")
plt.close()
print("Distribution of Scores saved")

# Descriptive Stats Q3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")
plt.savefig("assignments_01/outputs/boxplot.png")
plt.close()
print("Boxplot saved")

# Descriptive Stats Q4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Values")
plt.savefig("assignments_01/outputs/distribution_comparison.png")
plt.close()
print("Distribution Comparison saved")

# Descriptive Stats Q5
from scipy import stats

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print(f"Data1 Mean: {np.mean(data1)}")
print(f"Data1 Median: {np.median(data1)}")
print(f"Data1 Mode: {stats.mode(data1).mode}")

print(f"Data2 Mean: {np.mean(data2)}")
print(f"Data2 Median: {np.median(data2)}")
print(f"Data2 Mode: {stats.mode(data2).mode}")
# Data2 has an outlier (150) which pulls the mean up significantly.
# The median is not affected by outliers because it only looks
# at the middle value. That is why median and mean are so different for data2.

## ---- Hypothesis Testing ----

#Hypothesis Q1
from scipy import stats

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_value = stats.ttest_ind(group_a,group_b)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

#Hypothesis Q2

if p_value < 0.05:
    print(f"Result is statistically significant : p={p_value:.4f}")
else:
    print(f"Result is not statistically significant: p={p_value:.4f}")

#Hypothesis Q3
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat_new, p_value_new = stats.ttest_rel(before,after)
print(f"T-statistic: {t_stat_new}")
print(f"P-value: {p_value_new}")

#Hypothesis Q4
scores = [72, 68, 75, 70, 69, 74, 71, 73]
benchmark = 70

t_stat, p_value = stats.ttest_1samp(scores, benchmark)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

#Hypothesis Q5
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_value = stats.ttest_ind(group_a, group_b, alternative="less")
print(f"P-value (one-tailed): {p_value}")

#Hypothesis Q6
print("Group A scored significantly lower than Group B "
      "(mean A=71.5 vs mean B=82.1). "
      "This difference is unlikely due to chance (p < 0.05).")

# ---- Correlation ----

#Correlation Q1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr = np.corrcoef(x, y)
print("Correlation matrix:")
print(corr)
print(f"Correlation coefficient: {corr[0, 1]}")

# I expect correlation to be 1.0 because y = x * 2
# which is a perfect linear relationship

#Correlation Q2
from scipy.stats import pearsonr

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

corr, p_value = pearsonr(x, y)
print(f"Correlation coefficient: {corr}")
print(f"P-value: {p_value}")

#Correlation Q3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
corr = df.corr()
print(f"Correlation matrix: {corr}")

#Correlation Q4
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]


plt.scatter(x, y, color="blue")
plt.title("Negative Correlation")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("assignments_01/outputs/corr_plt_scatter.png")
plt.close()
print("Correlation scatter saved") 

#Correlation Q5
import seaborn as sns

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
corr = df.corr()

sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.savefig("assignments_01/outputs/correlation_heatmap.png")
plt.close()
print("Correlation Heatmap saved")

# ---- Pipelines ----

#Pipelines Q1
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    series = pd.Series(arr, name="values")
    return series

def clean_data(series):
    cleaned = series.dropna()
    return cleaned

def summarize_data(series):
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary

def data_pipeline(arr):
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    return summary

result = data_pipeline(arr)
for key, value in result.items():
    print(f"{key}: {value}")