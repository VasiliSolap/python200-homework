# --- LLMs as Transform ---

# Q1
# Parse "Jan 5th, 2024" into ISO format:
# Use code — standard format, datetime.strptime() handles it deterministically.

# Classify "my card was charged twice" into billing/technical/general:
# Use LLM — requires reading comprehension to interpret intent.

# Calculate the average of a list of numbers:
# Use code — pure arithmetic, deterministic.

# Extract company name from "Sr. Data Eng @ Acme Corp (contract)":
# Use LLM — freeform input with infinite format variations, no reliable regex pattern.

# Determine whether a review is more than 100 words:
# Use code — len(review.split()) > 100 is deterministic.

# Q2
# Problem: "Summarize in a few sentences" produces variable-length free text
# that cannot be reliably parsed or stored in a pipeline.
# Fixed prompt:
# system = (
#     "Summarize this product review in exactly one sentence. "
#     "Reply with that sentence only — no introduction, no punctuation at the end."
# )

# Q3
# 1. 50,000 records x 1 second = ~13.9 hours sequentially — too slow for production.
# 2. Use OpenAI Batch API — processes requests asynchronously at reduced cost,
#    without changing the model.

# --- Azure OpenAI ---

# Q1
# 1. Data residency: requests stay inside Azure infrastructure,
#    so sensitive data never leaves the organization's environment.
# 2. Unified billing: Azure OpenAI costs appear on the same Azure bill
#    as all other infrastructure, simplifying procurement.

# Q2
# azure_endpoint — the URL of the organization's Azure OpenAI resource,
#                  e.g. "https://mycompany.openai.azure.com"
# api_version    — the Azure OpenAI API version to use,
#                  e.g. "2024-02-01"
# api_key        -- the access key for the Azure OpenAI resource,
#                   issued by Azure (found in the resource's "Keys and Endpoint" section)

# Q3
# The model parameter takes a deployment name, not a model name like "gpt-4o-mini".
# The deployment name is configured by the infrastructure team and can be found
# in Azure AI Foundry under Deployments, or provided directly by your team.