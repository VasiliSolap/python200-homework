# project_09.py
# Video link: 

import json
import requests
import pandas as pd
from datetime import date
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from azure.storage.blob import BlobServiceClient, ContainerClient

ACCOUNT_URL = "https://vasilictd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

def extract():
    url = "https://api.open-meteo.com/v1/forecast?latitude=" \
    "35.2271&longitude=-80.8431&hourly=temperature_2m," \
    "precipitation&forecast_days=7"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def serialize(data):
    return json.dumps(data).encode("utf-8")

def load(blob_bytes):
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(ACCOUNT_URL, credential=credential)
    container_client = blob_service_client.get_container_client(CONTAINER)
    
    blob_path = f"raw/{date.today().isoformat()}/weather.json"
    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(blob_bytes, overwrite=True)
    
    print(f"Uploaded {len(blob_bytes)} bytes to {blob_path}")
    return container_client

def verify(container_client):
    print("\nBlobs in container:")
    for blob in container_client.list_blobs():
        print(f"{blob.name}  {blob.size} bytes")

def read_back(container_client):
    blob_path = f"raw/{date.today().isoformat()}/weather.json"
    blob_client = container_client.get_blob_client(blob_path)
    
    downloaded = blob_client.download_blob().readall()
    data = json.loads(downloaded.decode("utf-8"))
    
    df = pd.DataFrame(data['hourly'])
    print(df.head())
    
    with open("outputs/weather_raw.json", "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    data = extract()
    blob_bytes = serialize(data)
    container_client = load(blob_bytes)
    verify(container_client)
    read_back(container_client)