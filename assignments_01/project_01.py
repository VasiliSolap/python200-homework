import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from prefect import task, flow
from prefect.logging import get_run_logger
import os



DATA_DIR = "assignments_01/resources/happiness_project"
OUTPUT_DIR = "assignments_01/outputs"

@task(retries=3, retry_delay_seconds=2)
def load_data():
    logger = get_run_logger()  # ← скобки!
    all_data = []

    for year in range(2015, 2025):  # ← year не years
        file_path = f"{DATA_DIR}/world_happiness_{year}.csv"
        df = pd.read_csv(file_path, sep=";", decimal=",")
        df["year"] = year
        all_data.append(df)
        logger.info(f"Loaded year: {year}")
    
    merged = pd.concat(all_data, ignore_index=True)
    merged.to_csv(f"{OUTPUT_DIR}/merged_happiness.csv", index=False)
    logger.info(f"Total rows: {len(merged)}")
    return merged

@task
def descriptive_stats(df):
    logger = get_run_logger()
    
    mean = df["Happiness score"].mean()
    median = df["Happiness score"].median()
    std = df["Happiness score"].std()
    
    logger.info(f"Mean happiness: {mean}")
    logger.info(f"Median happiness: {median}")
    logger.info(f"STD happiness: {std}")

    by_year = df.groupby("year")["Happiness score"].mean()
    logger.info(f"By year:\n{by_year}")

    by_region = df.groupby("Regional indicator")["Happiness score"].mean()
    logger.info(f"By region:\n{by_region}")
    
    return df

@task
def visualizations(df):
    logger = get_run_logger()

#HISTOGRAMM
    plt.hist(df["Happiness score"], bins=20)
    plt.title("Happiness Score Distribution")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.savefig(f"{OUTPUT_DIR}/happiness_histogram.png")
    plt.close()
    logger.info("Saved happiness_histogram.png")

#BOXPLOT
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

#SCATTER PLOT
    x = df["GDP per capita"]
    y = df["Happiness score"]
    plt.scatter(x, y, alpha=0.5)
    plt.title("GDP vs Happiness Score")
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness Score")
    plt.savefig(f"{OUTPUT_DIR}/gdp_vs_happiness.png")
    plt.close()
    logger.info("Saved gdp_vs_happiness.png")

#CORRELATION HEATMAP
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

@task
def hypothesis_testing(df):
    logger = get_run_logger()
    
    #2019 vs 2020
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
    
    #Western Europe vs Sub-Saharan Africa
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
    
    number_of_tests = len(numeric_cols)
    adjusted_alpha = 0.05 / number_of_tests
    logger.info(f"Adjusted alpha (Bonferroni): {adjusted_alpha:.4f}")
    
    for col in numeric_cols:
        clean = df[["Happiness score", col]].dropna()
        corr, p_value = stats.pearsonr(clean["Happiness score"], clean[col])
        
        significant = "YES" if p_value < 0.05 else "NO"
        significant_bonferroni = "YES" if p_value < adjusted_alpha else "NO"
        
        logger.info(f"{col}: r={corr:.3f}, p={p_value:.4f}, "
                   f"significant={significant}, "
                   f"after Bonferroni={significant_bonferroni}")
    
    return df


@task
def summary_report(df):
    logger = get_run_logger()
    
   
    total_countries = df["Country"].nunique()
    total_years = df["year"].nunique()
    logger.info(f"Total countries: {total_countries}, Total years: {total_years}")
    
  
    by_region = df.groupby("Regional indicator")["Happiness score"].mean().sort_values(ascending=False)
    top3 = by_region.head(3)
    bottom3 = by_region.tail(3)
    logger.info(f"Top 3 regions:\n{top3}")
    logger.info(f"Bottom 3 regions:\n{bottom3}")
    
    # 2019 vs 2020
    logger.info("2019 vs 2020: No significant difference found (p=0.5953). "
                "Pandemic did not significantly affect global happiness in 2020.")
    
    
    logger.info("Strongest correlation with happiness: GDP per capita and Social support")
    
    return df

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