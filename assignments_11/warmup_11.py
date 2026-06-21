# --- Prefect Orchestration ---

# Prefect Q1

# @task is a single unit of work (API call, file upload, data transform).
# Prefect tracks its state, logs it, and can retry it on failure.
#
# @flow is the orchestrator -- it calls tasks in order and manages
# the pipeline as a whole.
#
# Should celsius_to_fahrenheit be decorated with @task? No.
# @task is useful when there is external I/O that can fail (API calls,
# Blob Storage, database). A pure in-memory calculation like
# (temp * 9/5) + 32 will never fail due to external reasons,
# so retries and state tracking add no value here.


# Prefect Q2

# @task(retries=3, retry_delay_seconds=30)
# def call_api():


# Prefect Q3

# Click on the failed flow run, then click on the Failed transform task.
# Open the Logs tab -- there I will find the exact error and traceback.
# load never ran because transform failed first.


# --- Production Patterns ---


# Production Q1

# raise_for_status() raises an exception immediately if the API returns
# an error (4xx or 5xx). This stops the task right there and Prefect
# marks it as Failed.
#
# if response.status_code != 200: print("error") just prints a message
# and continues -- the broken response gets passed to transform(), which
# then fails with a confusing error (KeyError, TypeError, etc).
# The real problem was in extract(), but it looks like transform() broke.


# Production Q2

# Run #1: transform() crashes halfway, load() never runs.
# The blob does not exist yet (or still has old data if it existed before).
#
# Run #2 (after fixing the bug):
# With overwrite=True:
# load() uploads the new file and overwrites any existing blob. Safe to re-run.
#
# Without overwrite=True:
# load() raises ResourceExistsError because the blob already exists.
# The pipeline fails even though the data is correct and the bug is fixed.
# You would have to manually delete the blob before every re-run.


# Production Q3

from prefect.logging import get_run_logger

@task
def load(records: list, blob_path: str) -> None:
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records to {blob_path}")