import os
from pathlib import Path
from dotenv import load_dotenv
from scipy import stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from smolagents import CodeAgent, OpenAIServerModel, tool

if load_dotenv():
    print('Successfully loaded environment variables from .env')

api_key = os.getenv("OPENAI_API_KEY")

DATA_PATH = "resources/merged_happiness.csv"

df = None


# ---Task 1: Define Tools---

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Returns:
        A dict with shape and columns of the loaded dataset.
    """
    global df
    df = pd.read_csv(DATA_PATH)
    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }

@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column.

    Args:
        column: The name of the column to summarize.

    Returns:
        A dict with descriptive statistics.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    
    return df[column].describe().to_dict()  # ← как получить статистику колонки?

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two columns.

    Args:
        col1: First column name.
        col2: Second column name.

    Returns:
        A dict with col1, col2, pearson_r, and p_value.
    """
    if df is None:
        return {"error": "No data loaded."}
    
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": f"Column not found."}
    
    pearson_r, p_value = stats.pearsonr(df[col1], df[col2])
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(pearson_r, 4),
        "p_value": round(p_value, 4)
    }

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The column to rank by.
        year: The year to filter by.
        n: Number of top countries to return. Defaults to 5.

    Returns:
        A dict with a list of top countries and their values.
    """
    if df is None:
        return {"error": "No data loaded."}
    
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    
    year_df = df[df['year'] == year]
    
    if year_df.empty:
        return {"error": f"No data for year {year}."}
    
    top = year_df.sort_values(column, ascending=False).head(n)
    
    return {
        "results": top[['Country', column]].to_dict('records')
    }



# Task 2 — Build the Agent
SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


# Task 3 — Run Guided Queries
if __name__ == "__main__":
    
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

# Task 4 — My Own Questions

    # My query 1
    my_query_1 = "What are the top 3 happiest countries in 2019?"
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: This triggered tool use — get_top_n_countries was called
    # with year=2019 and n=3. No code generation needed.

    # My query 2
    my_query_2 = "Plot a histogram of happiness scores and save it to outputs/happiness_histogram.png"
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: This triggered code generation — no tool covers histogram plots.
    # The agent wrote matplotlib code directly.

# --- Reflection ---
#
# 1. In Query 3, the agent used p_value < 0.05 to check significance.
#    The p_value was 0.0, so it correctly said the correlation is significant.
#
# 2. Query 4 surprised me — the agent found the top 5 countries in one step.
#    But Query 5 disappointed — it used dummy data instead of real data for
#    the plot because it couldn't access the DataFrame directly.
#
# 3. A filter_by_region tool would help. It would return data for a specific
#    region so the agent could answer "Which region is happiest overall?"