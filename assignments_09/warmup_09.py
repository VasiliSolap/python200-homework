from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import SubscriptionClient

# --- Azure Authentication ---


# Azure Authentication Question 1

# Locally, DefaultAzureCredential relies on the az login session.
# You must run `az login` in the terminal first — it saves a token
# to ~/.azure/ on your machine.
# DefaultAzureCredential automatically checks that folder and uses
# the token if it finds one. No extra code needed.


# Azure Authentication Question 2

# A deployed pipeline cannot use az login because there is no human
# present to open a browser and enter credentials. az login is an
# interactive process — it requires a person.
# Instead, it uses Managed Identity — an identity that Azure
# automatically assigns to the VM or container. Azure generates
# and rotates the token internally, no human required.
# The same Python code works without changes because
# DefaultAzureCredential tries multiple authentication methods
# in order. Locally it finds the az login token, on the server
# it finds the Managed Identity — the code itself never changes.


# Azure Authentication Question 3

# Cause 1: Azure CLI is not installed
# How to diagnose: run az --version in the terminal
# if you see "command not found" — CLI is not installed
# Fix: brew install azure-cli
#
# Cause 2: az login was never run
# How to diagnose: run az account show in the terminal
# if you see "Please run az login" — token does not exist
# Fix: run az login


# --- Blob Storage ---

# Blob Storage Question 1

# Storage Account — top level resource in Azure, holds everything inside it
# Container — a named bucket inside the Storage Account,
# used to organize blobs (like a folder)
# Blob — an individual file stored inside a container,
# can be any format: CSV, JSON, image, binary, etc.
# Analogy:
# Storage Account — like Google Drive (the entire storage)
# Container — like a folder inside Google Drive
# Blob — like a file inside the folder


# Blob Storage Question 2

# Scenario 1: REST API JSON payload each hour
# Use Blob Storage because we are storing raw files
# that don't need to be queried or filtered

# Scenario 2: 50 million customer transactions
# Use Azure SQL because the analytics team queries
# by date range and customer ID every day

# Scenario 3: NumPy arrays of image embeddings
# Use Blob Storage because we are simply saving
# and loading files between pipeline runs,
# no filtering needed


# Blob Storage Question 3

def list_container(container_client):
   for blob in container_client.list_blobs():
        print(f"{blob.name}  {blob.size} bytes")


# Blob Storage Question 4

def upload_text(container_client, blob_name, text):
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(text.encode("utf-8"), overwrite=True)

    