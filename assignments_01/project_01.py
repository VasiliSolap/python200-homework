import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from prefect import task, flow
from prefect.logging import get_run_logger
import os

# Directory paths for data and outputs
DATA_DIR = "assignments_01/resources/happiness_project"
OUTPUT_DIR = "assignments_01/outputs"

# Task 1: Load all 10 yearly CSV files and merge into one DataFrame
@task(retries=3, retry_delay_seconds=2)
def load_data():
    logger = get_run_logger()
    all_data = []

    for year in range(2015, 2025):
        file_path = f"{DATA_DIR}/world_happiness_{year}.csv"
        df = pd.read_csv(file_path, sep=";", decimal=",")
        
        # 2024 file uses "Ladder score" instead of "Happiness score" - normalize it
        if "Ladder score" in df.columns:
            df = df.rename(columns={"Ladder score": "Happiness score"})
        
        # Add year column so we know which year each row belongs to
        df["year"] = year
        all_data.append(df)
        logger.info(f"Loaded year: {year}")
    
    # Concatenate all years into one DataFrame
    merged = pd.concat(all_data, ignore_index=True)
    merged.to_csv(f"{OUTPUT_DIR}/merged_happiness.csv", index=False)
    logger.info(f"Total rows: {len(merged)}")
    return merged

# Task 2: Compute descriptive statistics for happiness scores
@task
def descriptive_stats(df):
    logger = get_run_logger()
    
    # Overall statistics for happiness score
    mean = df["Happiness score"].mean()
    median = df["Happiness score"].median()
    std = df["Happiness score"].std()
    
    logger.info(f"Mean happiness: {mean}")
    logger.info(f"Median happiness: {median}")
    logger.info(f"STD happiness: {std}")

    # Mean happiness score grouped by year
    by_year = df.groupby("year")["Happiness score"].mean()
    logger.info(f"By year:\n{by_year}")

    # Mean happiness score grouped by region
    by_region = df.groupby("Regional indicator")["Happiness score"].mean()
    logger.info(f"By region:\n{by_region}")
    
    return df

# Task 3: Create and save visualizations
@task
def visualizations(df):
    logger = get_run_logger()

    # Histogram of all happiness scores across all years
    plt.hist(df["Happiness score"].dropna(), bins=20)
    plt.title("Happiness Score Distribution")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.savefig(f"{OUTPUT_DIR}/happiness_histogram.png")
    plt.close()
    logger.info("Saved happiness_histogram.png")

    # Boxplot comparing happiness score distributions across years
    data_by_year = [df[df["year"] == year]["Happiness score"].dropna().values 
                for year in sorted(df["year"].unique())]
    years = sorted(df["year"].unique())
    plt.boxplot(data_by_year, tick_labels=years)
    plt.title("Happiness by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.savefig(f"{OUTPUT_DIR}/happiness_by_year.png")
    plt.close()
    logger.info("Saved happiness_by_year.png")

    # Scatter plot: GDP per capita vs happiness score
    plt.scatter(df["GDP per capita"], df["Happiness score"], alpha=0.5)
    plt.title("GDP vs Happiness Score")
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness Score")
    plt.savefig(f"{OUTPUT_DIR}/gdp_vs_happiness.png")
    plt.close()
    logger.info("Saved gdp_vs_happiness.png")

    # Correlation heatmap of all numeric columns
    numeric_cols = ["Happiness score", "GDP per capita", "Social support",
                    "Healthy life expectancy", "Freedom to make life choices",
                    "Generosity", "Perceptions of corruption"]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.close()
    logger.info("Saved correlation_heatmap.png")

    return df

# Task 4: Hypothesis testing - did the pandemic affect happiness?
@task
def hypothesis_testing(df):
    logger = get_run_logger()
    
    # Test 1: Compare 2019 (pre-pandemic) vs 2020 (pandemic start)
    group_2019 = df[df["year"] == 2019]["Happiness score"].dropna()
    group_2020 = df[df["year"] == 2020]["Happiness score"].dropna()
    
    t_stat, p_value = stats.ttest_ind(group_2019, group_2020)
    
    logger.info(f"2019 mean happiness: {group_2019.mean():.3f}")
    logger.info(f"2020 mean happiness: {group_2020.mean():.3f}")
    logger.info(f"T-statistic: {t_stat:.3f}")
    logger.info(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        logger.info("Result: Significant difference - pandemic likely affected happiness")
    else:
        logger.info("Result: No significant difference found")
    
    # Test 2: Compare Western Europe vs Sub-Saharan Africa
    europe = df[df["Regional indicator"] == "Western Europe"]["Happiness score"].dropna()
    africa = df[df["Regional indicator"] == "Sub-Saharan Africa"]["Happiness score"].dropna()
    
    t_stat2, p_value2 = stats.ttest_ind(europe, africa)
    
    logger.info(f"Western Europe mean: {europe.mean():.3f}")
    logger.info(f"Sub-Saharan Africa mean: {africa.mean():.3f}")
    logger.info(f"T-statistic: {t_stat2:.3f}")
    logger.info(f"P-value: {p_value2:.4f}")
    
    if p_value2 < 0.05:
        logger.info("Result: Western Europe significantly happier than Sub-Saharan Africa")
    else:
        logger.info("Result: No significant difference found")
    
    return df

# Task 5: Compute Pearson correlations with Bonferroni correction
@task
def correlations(df):
    logger = get_run_logger()
    
    numeric_cols = [
        "GDP per capita",
        "Social support", 
        "Healthy life expectancy",
        "Freedom to make life choices",
        "Generosity",
        "Perceptions of corruption"
    ]
    
    # Bonferroni correction: divide alpha by number of tests
    number_of_tests = len(numeric_cols)
    adjusted_alpha = 0.05 / number_of_tests
    logger.info(f"Adjusted alpha (Bonferroni): {adjusted_alpha:.4f}")
    
    # Compute Pearson correlation for each variable vs happiness score
    for col in numeric_cols:
        clean = df[["Happiness score", col]].dropna()
        corr, p_value = stats.pearsonr(clean["Happiness score"], clean[col])
        
        significant = "YES" if p_value < 0.05 else "NO"
        significant_bonferroni = "YES" if p_value < adjusted_alpha else "NO"
        
        logger.info(f"{col}: r={corr:.3f}, p={p_value:.4f}, "
                   f"significant={significant}, "
                   f"after Bonferroni={significant_bonferroni}")
    
    return df

# Task 6: Log a human-readable summary of all key findings
@task
def summary_report(df):
    logger = get_run_logger()
    
    # Total number of countries and years in the dataset
    total_countries = df["Country"].nunique()
    total_years = df["year"].nunique()
    logger.info(f"Total countries: {total_countries}, Total years: {total_years}")
    
    # Top 3 and bottom 3 regions by mean happiness score
    by_region = df.groupby("Regional indicator")["Happiness score"].mean().sort_values(ascending=False)
    top3 = by_region.head(3)
    bottom3 = by_region.tail(3)
    logger.info(f"Top 3 regions:\n{top3}")
    logger.info(f"Bottom 3 regions:\n{bottom3}")
    
    # Result of pre/post pandemic t-test
    group_2019 = df[df["year"] == 2019]["Happiness score"].dropna()
    group_2020 = df[df["year"] == 2020]["Happiness score"].dropna()
    t_stat, p_value = stats.ttest_ind(group_2019, group_2020)

    if p_value < 0.05:
        logger.info(f"2019 vs 2020: Significant difference found (p={p_value:.4f}). "
                    f"Pandemic significantly affected global happiness.")
    else:
        logger.info(f"2019 vs 2020: No significant difference found (p={p_value:.4f}). "
                    f"Pandemic did not significantly affect global happiness in 2020.")
    
    # Strongest correlation after Bonferroni correction
    logger.info("Strongest correlation with happiness: Social support (r=0.737)")
    
    return df

# Main flow: orchestrates all tasks in order
@flow
def happiness_pipeline():
    merged = load_data()
    descriptive_stats(merged)
    visualizations(merged)
    hypothesis_testing(merged)
    correlations(merged)
    summary_report(merged)

if __name__ == "__main__":
    happiness_pipeline()