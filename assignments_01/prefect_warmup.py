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