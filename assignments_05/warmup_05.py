# --- The Chat Completions API ---

#API Q1

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("API Q1 — Response:", response.choices[0].message.content)
print("API Q1 — Model:", response.model)
print("API Q1 — Total tokens:", response.usage.total_tokens)

# API Q2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"API Q2 — Temperature {temp}: {response.choices[0].message.content}")


# At temperature=0 the output is always the same — fully deterministic.
# At temperature=0.7 there is a balance between creativity and consistency.
# At temperature=1.5 the output is highly random and unpredictable.

# API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices):
    print(f"API Q3 — Choice {i+1}: {choice.message.content}")

# API Q4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)

print(f"API Q4 — Response: {response.choices[0].message.content}")

# The response was cut off mid-sentence because max_tokens=15
# limits the total number of tokens in the response.
# In real applications, max_tokens is useful for controlling costs,
# keeping responses concise, and preventing unexpectedly long outputs.


# --- System Messages and Personas ---

# System Q1 — Personality 1: Python tutor

messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print(f"System Q1 — Tutor: {response.choices[0].message.content}")

# System Q1 — Personality 2: Grumpy senior developer
messages2 = [
    {"role": "system", "content": "You are a grumpy senior developer who thinks everyone should just read the documentation. You give correct but very blunt answers."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages2
)
print(f"System Q1 — Grumpy dev: {response2.choices[0].message.content}")

# The system message completely changes the tone and style of the response.
# The tutor is warm and encouraging, the grumpy dev is blunt and dismissive.
# Same question, same model — but totally different personality and output.

# System Q2

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print(f"System Q2 — Response: {response.choices[0].message.content}")

# The model knows Jordan's name because we manually passed the full
# conversation history in the messages list. Even though the API is stateless
# and has no memory between calls, we simulate memory by including all previous
# messages in every request. The model simply reads the context we provide.

# --- Prompt Engineering ---

# Prompt Q1 — Zero-Shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews):
    prompt = f"Classify the sentiment of this review as positive, negative or mixed:\n\n{review}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user", "content": prompt}],
        temperature=0
    )

print(f"Prompt Q1 - Review{i+1}: {response.choices[0].message.content}")

# Prompt Q2 — One-Shot

for i, review in enumerate(reviews):
    prompt = f"""Classify the sentiment of this review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify:
Review: "{review}"
Sentiment:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    print(f"Prompt Q2 — Review {i+1}: {response.choices[0].message.content}")

# Adding one example made the output more consistent and concise.
# Instead of a full sentence explanation, the model now returns just one word
# matching the format shown in the example.


# Prompt Q3 — Few-Shot

for i, review in enumerate(reviews):
    prompt = f"""Classify the sentiment of this review as positive, negative, or mixed.

Example 1:
Review: "The team was incredibly helpful and resolved my issue immediately."
Sentiment: positive

Example 2:
Review: "The product broke after one day and customer service ignored my emails."
Sentiment: negative

Example 3:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify:
Review: "{review}"
Sentiment:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    print(f"Prompt Q3 — Review {i+1}: {response.choices[0].message.content}")


# Zero-shot: easiest to write but inconsistent format — model explains instead of labeling.
# One-shot: better consistency, but occasional format drift (e.g. repeating "Sentiment:").
# Few-shot: most consistent and reliable — multiple examples lock in the exact format.
# Use zero-shot for simple tasks, one-shot when format matters,
# few-shot when consistency and accuracy are critical.

# Prompt Q4 — Chain of Thought

prompt = """Show your step-by-step reasoning, then give the final answer on its own line labeled: Final answer: <value>

Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)
print(f"Prompt Q4 — Chain of Thought:\n{response.choices[0].message.content}")

# Asking the model to reason step by step improves accuracy because
# it forces the model to break the problem into smaller parts before committing
# to a final answer. This reduces the chance of skipping a calculation or
# making an arithmetic error. 

# Prompt Q5 — Structured Output
import json

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = f"""Analyze the review below and respond ONLY with valid JSON, no other text.
Keys: sentiment, confidence (a float from 0 to 1), reason (one sentence).

Review:
\"\"\"{review}\"\"\"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

raw = response.choices[0].message.content
print(f"Prompt Q5 — Raw response: {raw}")

try:
    result = json.loads(raw)
    print(f"Prompt Q5 — Sentiment: {result['sentiment']}")
    print(f"Prompt Q5 — Confidence: {result['confidence']}")
    print(f"Prompt Q5 — Reason: {result['reason']}")
except json.JSONDecodeError:
    print(f"Prompt Q5 — Error: response was not valid JSON")
    print(f"Prompt Q5 — Raw output: {raw}")


# Prompt Q6 — Delimiters

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

````{user_text}```"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)
print(f"Prompt Q6 — With instructions:\n{response.choices[0].message.content}")

# Test with non-instruction text
plain_text = "The weather in Paris is lovely in the spring."

prompt2 = f"""You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{plain_text}```"""

response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt2}],
    temperature=0
)
print(f"Prompt Q6 — Without instructions: {response2.choices[0].message.content}")

# Delimiters clearly separate user data from instructions, preventing
# prompt injection — a situation where user input accidentally overrides
# or interferes with the instructions given to the model.
# They make the prompt more predictable and easier to parse programmatically.


# --- Local Models with Ollama ---

# Ollama Q1
# Ollama CLI output (run manually in terminal):
# Command: ollama run qwen3:0.6b "Explain what a large language model is in two sentences."
# Output:
"""
A large language model is an AI system trained on vast amounts of text data,
enabling it to understand and generate human-like language. It processes and
interprets complex information, allowing it to perform tasks like writing,
answering questions, or creating content with remarkable accuracy.
"""

# OpenAI API — same prompt
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what a large language model is in two sentences."}],
    temperature=0
)
print(f"Ollama Q1 — OpenAI response: {response.choices[0].message.content}")

# Comment: Differences observed:
# - qwen3:0.6b showed its internal "thinking" process before the final answer,
#   while gpt-4o-mini gave a clean, direct response.
# - gpt-4o-mini produced a more polished and concise answer.
# - qwen3:0.6b sometimes repeated itself and was less focused.
#
# Advantage of running locally: free, private, no API costs, works offline.
# Disadvantage: smaller models are less accurate and slower on CPU,
# requires local storage (~500MB+) and setup.