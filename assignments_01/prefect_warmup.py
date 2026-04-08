from prefect import task, flow
import numpy as np
import pandas as pd

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    series = pd.Series(arr, name="values")
    return series

@task
def clean_data(series):
    cleaned = series.dropna()
    return cleaned

@task
def summarize_data(series):
    summary = {
        "mean":   series.mean(),
        "median": series.median(),
        "std":    series.std(),
        "mode":   series.mode()[0]
    }
    return summary

@flow
def pipeline_flow():
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    pipeline_flow()

# 1.Why might Prefect be more overhead than it is worth here?
# This pipeline is simple - just three small functions on a handful of numbers.
# It runs in under a second and never fails. Starting a server, logging every step,
# and tracking task states is too much overhead for such a small task.

# 2. Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline 
# logic itself stays simple like in this case.
# - When the pipeline runs automatically every day on a schedule
# - When data is loaded from the internet and the connection might fail
# - When the pipeline runs on a server and you need to know what happened
# - When you need to know exactly which step failed and why
