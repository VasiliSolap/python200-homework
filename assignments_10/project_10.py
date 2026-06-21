
## Video : https://youtu.be/AtxYrMnPV4s

import json
import os
from datetime import date
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

load_dotenv()

ACCOUNT_URL = "https://vasilictd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
VALID_LABELS = {"good", "marginal", "bad"}
SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

# Weather classification is a borderline LLM use case.
# Temperature and precipitation are numbers — deterministic code could handle this:
# if temperature > 10 and precipitation < 1: return "good".
# You would lose flexibility but gain speed, lower cost, and full predictability.


# --- Step 1: Read ---

credential = DefaultAzureCredential()
container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)

blob_path = "raw/2026-06-02/weather.json"
raw = container.download_blob(blob_path).readall()
data = json.loads(raw.decode("utf-8"))

hourly = data["hourly"]
records = []
for i in range(len(hourly["time"])):
    records.append({
        "time": hourly["time"][i],
        "temperature_2m": hourly["temperature_2m"][i],
        "precipitation": hourly["precipitation"][i],
    })

print(f"Loaded {len(records)} records")

# --- Step 2: Transform ---

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
enriched = []

for i, record in enumerate(records[:24]):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Temperature: {record['temperature_2m']}C, "
                    f"Precipitation: {record['precipitation']}mm"
                ),
            },
        ],
    )
    raw_label = response.choices[0].message.content.strip().lower()
    label = raw_label if raw_label in VALID_LABELS else "unknown"
    enriched.append({**record, "conditions": label})

    if (i + 1) % 6 == 0:
        print(f"  Processed {i + 1} records...")

# --- Step 3: Write ---

today = date.today().isoformat()
processed_path = f"processed/{today}/weather_classified.json"
container.upload_blob(
    processed_path,
    json.dumps(enriched).encode("utf-8"),
    overwrite=True,
)
print(f"Uploaded to {processed_path}")

# --- Step 4: Spot-Check ---

raw_processed = container.download_blob(processed_path).readall()
df = pd.DataFrame(json.loads(raw_processed.decode("utf-8")))

print("\nLabel distribution:")
print(df["conditions"].value_counts())
print("\nFirst 5 rows:")
print(df.head(5))

# --- Step 5: Save Output ---

os.makedirs("outputs", exist_ok=True)
with open("outputs/first_10_records.json", "w") as f:
    json.dump(enriched[:10], f, indent=2)

print("\nSaved first 10 records to outputs/first_10_records.json")