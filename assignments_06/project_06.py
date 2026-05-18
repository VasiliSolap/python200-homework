
# --- Step 1: Setup ---
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from pathlib import Path
docs_dir = Path("groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

SimpleDirectoryReader("groundwork_docs")
Settings.llm = LlamaOpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


# --- Step 2: Load the Documents ---

documents = SimpleDirectoryReader("groundwork_docs").load_data()
print(f"Loaded {len(documents)} documents")

for doc in documents:
    print(doc.metadata["file_name"])


# --- Step 3: Build the Index and Query Engine ---

index = VectorStoreIndex.from_documents(documents)
print("Index built successfully. Ready to answer questions.")

# Build query engine
query_engine = index.as_query_engine(similarity_top_k=3)


# --- Step 4: Query the Assistant ---

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    top_node = response.source_nodes[0]
    print(f"\nTop Source:")
    print(f"  File:  {top_node.node.metadata['file_name']}")
    print(f"  Score: {top_node.score:.4f}")
    print(f"  Text:  {top_node.node.get_content()[:200]}")

# The assistant answered all five questions accurately and confidently.
#
# Surprising observations:
# Q1 (weekend hours): correct answer but top source was our_story.txt
#    not faq.txt — the hours information was retrieved from
#    a less obvious document but the answer was still correct.
#
# Q2 (dairy-free options): correct answer but top source was
#    seasonal_specials.txt not menu.txt — because seasonal drinks
#    also mention dairy-free options prominently.
#
# Q3 (loyalty program): low score (0.2822) but correct answer —
#    shows that even low similarity scores can produce good responses.
#
# Q4 (company history): highest score (0.7569) and most detailed
#    response — our_story.txt was a perfect match.
#
# Q5 (catering/wholesale): correct source and confident answer.
#
# Overall: the assistant sounded confident and accurate throughout.
# RAG successfully grounded all responses in the actual documents.


# --- Step 5: Find a Failure ---

q_hard = "What is the calorie count of the croissant?"

print(f"Q: {q_hard}")
response = query_engine.query(q_hard)
print(f"A: {response}")

# Print ALL three source nodes
for i, node in enumerate(response.source_nodes):
    print(f"\nNode {i+1}:")
    print(f"  File:  {node.node.metadata['file_name']}")
    print(f"  Score: {node.score:.4f}")
    print(f"  Text:  {node.node.get_content()[:200]}")

# Query: "What is the calorie count of the croissant?"
# Expected to be hard because calorie information
# is not in any of the documents.
#
# What happened:
# menu.txt was retrieved first (score: 0.2845) which is correct
# but the score was very low — weak retrieval.
# The model honestly admitted it could not find the answer.
#
# Tone: the model became uncertain and did not guess.
# This is safe behavior but not guaranteed —
# sometimes the model sounds confident even when retrieval fails.
#
# To improve: add a system prompt telling the model
# to only answer from documents and say "I don't know"
# when information is missing.


# --- Step 6: Reflection ---
#
# 1. LlamaIndex implementation took about 5 lines of code:
#    Settings, SimpleDirectoryReader, VectorStoreIndex,
#    and as_query_engine. The manual RAG implementation
#    from the lesson required dozens of lines for chunking,
#    embedding, and indexing. Frameworks hide complexity
#    and let developers focus on the actual problem.
#
# 2. A hospital could use RAG to build an assistant
#    for medical staff. Documents would include drug
#    instructions, treatment protocols, and clinical studies.
#    Doctors could ask questions like "What is the correct
#    dosage for children?" and get answers directly from
#    official documentation without searching hundreds of pages.
#    This reduces errors and saves time in critical situations.
#
# 3. One failure mode RAG cannot prevent:
#    when the answer is simply not in the documents.
#    Even when retrieval works correctly and finds the most
#    relevant chunk, if the information was never written down
#    the model cannot answer accurately.
#    In our project, menu.txt was correctly retrieved for
#    the calorie question — but calorie data was never there.
#    RAG can only work with what exists in the documents.