from dotenv import load_dotenv
import os
import string
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# --- RAG Concepts ---


# Concepts Q1

# Scenario A: RAG
# The legal team has hundreds of PDFs that are updated every quarter.
# Fine-tuning would need to be repeated every update which is costly and slow.
# RAG allows the system to search only the relevant sections at query time
# without retraining the model.

# Scenario B: Fine-tuning
# The goal is a specific brand voice that does not appear much online,
# so the base model has never seen this style.
# With 3,000 examples from in-house writers, fine-tuning teaches the model
# exactly how this brand writes.

# Scenario C: Prompt Engineering
# The document is only two pages — it fits easily into a single prompt.
# The task is one-time only, so building a RAG pipeline would be overkill.
# Simply paste the document directly into the prompt.


# Concepts Q2

# A confidently wrong answer is more harmful because we are wired to trust
# certainty. When a model says "I am not sure", we naturally go and verify.
# When it sounds confident, we accept it without checking.
#
# Real example: A doctor asks an AI about the dosage of a new medication.
# The prompt was slightly ambiguous — the doctor meant one drug but the model
# answered about another. The AI confidently responds: "According to 2023
# clinical protocols, the standard adult dosage is 2ml, confirmed by WHO
# research." The doctor trusts the answer, administers the wrong dose,
# and the patient is harmed.
#
# Tone matters because confident language signals reliability.
# When a response includes specific references like "clinical protocols" or
# "WHO research" — even if invented — it sounds authoritative.
# We trust authoritative voices without questioning them.
# The more convincing the hallucination sounds, the more dangerous it is.


# Concepts Q3

# Correct order of RAG pipeline steps:
# 1. Extract text from source documents
#    — raw text is pulled from PDFs, Word files, etc.
# 2. Split text into chunks
#    — long documents are broken into smaller pieces for retrieval
# 3. Convert text chunks into embeddings
#    — each chunk is turned into a vector representing its meaning
# 4. Receive the user's query
#    — the user asks a question
# 5. Embed the user's query
#    — the query is also converted into a vector
# 6. Retrieve the most relevant chunks
#    — cosine similarity finds the chunks closest to the query vector
# 7. Inject retrieved chunks into the prompt
#    — the chunks are added to the LLM prompt as context
# 8. Generate a response from the LLM
#    — the model answers using only the provided context


# --- Keyword RAG ---

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
    

#Keyword Q1

query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result = simple_keyword_retrieval(query, documents, verbose=True)
print("\nKeyword Q1")
print(f"\nSelected document: {result[0][0]}")


# loyalty.txt was selected — which is WRONG.
# "your" is not in the stopwords list so it matched
# both hiring.txt and loyalty.txt with overlap=1.
# "hours" appears in the filename but NOT in the document text,
# so hours.txt scored 0.
# This shows a key weakness of keyword RAG:
# stopword lists must be complete, and filenames
# are invisible to the retrieval function.


#Keyword Q2

query = "Do you have anything without caffeine?"

result = simple_keyword_retrieval(query, documents, verbose=True)
print("\nKeyword Q2")
print(f"\nSelected document: {result[0][0]}")

# None found — keyword RAG completely failed here.
# The word "caffeine" does not appear in any document.
# menu.txt contains coffee drinks (espresso, lattes, cold brew)
# which all contain caffeine — but keyword RAG cannot make
# this logical connection.
# It only matches exact words, not meaning or concepts.
#
# Semantic RAG would do much better here:
# it would convert "caffeine" into a vector and find that
# "espresso", "lattes", "cold brew" are semantically related,
# correctly retrieving menu.txt as the best match.

#Keyword Q3

# Prediction: No document will be selected because
# the word "rewards" does not appear in any document.
# loyalty.txt talks about "loyalty program" and "points"
# but not "rewards" — keyword RAG cannot match synonyms.

query = "How do I sign up for rewards?"

result = simple_keyword_retrieval(query, documents, verbose=True)
print("\nKeyword Q3")
print(f"\nSelected document: {result[0][0]}")

# Result: Prediction was CORRECT — None found.
# loyalty.txt was the obvious answer but scored 0
# because it never uses the word "rewards".
# "sign up" also had no matches anywhere.
#
# This confirms the core weakness of keyword RAG:
# it cannot understand that "rewards" = "loyalty program"
# Semantic RAG would easily find loyalty.txt because
# "rewards" and "loyalty program" have similar vector embeddings.


# --- Semantic RAG Concepts ---

#Semantic Q1

# 1. A vector embedding is a numerical representation of text —
# a list of numbers that captures the meaning of that text.
# Words or sentences with similar meanings end up with similar
# numbers, placing them close together in vector space.


# 2. Chunk A (0.85) is more relevant than Chunk B (0.30).
# Cosine similarity works like a percentage of meaning overlap:
# 0.85 means the texts are very close in meaning (85% similar),
# while 0.30 means they share little semantic relationship.
# The higher the score, the closer the two texts sit
# on the vector map — and the more relevant the chunk is.

# 3. Semantic search finds relevant chunks even without exact word matches
# because it compares meaning, not letters.
# For example: "caffeine" and "espresso" are different words
# but they always appear together in real-world text —
# so their vectors end up close on the map.
# The model learned their relationship from billions of texts,
# so searching for "caffeine" naturally leads to
# coffee-related content even without the exact word.


#Semantic Q2

# | Feature                 | Keyword RAG                    | Semantic RAG                    |
# |-------------------------|--------------------------------|---------------------------------|
# | What is compared?       | Exact word overlap             | Meaning via vector similarity   |
# | What is retrieved?      | Full document                  | Most relevant chunks            |
# | Can it handle synonyms? | No                             | Yes                             |
# | Storage format          | Plain text dictionary          | Vector store / index            |
# | Relevance score         | Number of overlapping keywords | Cosine similarity score         |


# --- LlamaIndex ---

SimpleDirectoryReader("06_AI_augmentation")

# LlamaIndex Q1

# Configure settings
Settings.llm = LlamaOpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Load documents and build index
documents = SimpleDirectoryReader("06_AI_augmentation").load_data()
print("\n LlamaIndex Q1")
print(f"Loaded {len(documents)} documents")

index = VectorStoreIndex.from_documents(documents)
print("Index built successfully!")

# Build query engine
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:100]}...")
        print("-" * 30)

# What employee benefits does BrightLeaf offer?
# Node 1 (0.7408) was highly relevant — correctly retrieved employee_benefits.pdf
# Nodes 2 and 3 were less relevant (partnerships, mission statement)
# The model responded confidently with specific details: Blue Cross, $600 wellness
# reimbursement, 401(k) match — no hedging phrases like "I am not sure"
# Nothing unexpected was retrieved for this query.

# What are BrightLeaf's security policies?
# Node 1 (0.6481) correctly retrieved security_policy.pdf
# Nodes 2 and 3 were less relevant (benefits, mission statement)
# The model responded with very specific technical details:
# TLS 1.3, AES-256, NIST 800-61 — sounded confident and authoritative
# Slightly unexpected: employee_benefits.pdf appeared as Node 2
# possibly because both documents mention employee roles and responsibilities


# --- LlamaIndex Q2 ---

q = "What employee benefits does BrightLeaf offer?"
print("\nLlamaIndex Q2")
for k in [1, 5]:
    print(f"\n{'='*60}")
    print(f"similarity_top_k={k}")
    qe = index.as_query_engine(similarity_top_k=k)
    response = qe.query(q)
    print(f"Answer: {response}")
    print("\nSource nodes:")
    for node in response.source_nodes:
        print(f"  Score: {node.score:.4f}")
        print(f"  Text: {node.node.get_content()[:100]}")
        print("-" * 30)

# top_k=1 produced a cleaner, more structured response with only
# the most relevant chunk (score: 0.7408, employee_benefits.pdf)
#
# top_k=5 added 4 irrelevant chunks (partnerships, mission, earnings,
# security) but the response quality did not drop significantly —
# the model correctly focused on the benefits chunk and ignored the rest.
#
# More context is NOT always better:
# - It costs more tokens
# - Irrelevant chunks can confuse the model on harder questions
# - If two chunks contradict each other, the model may hallucinate
# - For this simple factual query the difference was minimal,
#   but for complex multi-document reasoning top_k matters a lot.


# --- LlamaIndex Q3 ---

# Prediction: "Tell me everything about BrightLeaf" is too vague.
# "everything" has no clear vector direction — the retriever
# will pick only 3 random chunks and ignore the rest.
# The response will be incomplete and may miss entire topics.

q3 = "Tell me everything about BrightLeaf"
print("\nLlamaIndex Q3")
print(f"\n{'='*60}")
print(f"Q: {q3}")
qe3 = index.as_query_engine(similarity_top_k=3)
response3 = qe3.query(q3)
print(f"A: {response3}")
print("\nSource nodes:")
for node in response3.source_nodes:
    print(f"  Score: {node.score:.4f}")
    print(f"  Text: {node.node.get_content()[:150]}")
    print("-" * 30)

# Expected: incomplete or confused response because
# "everything" is too vague — no clear vector direction.
# top_k=3 can only retrieve 3 out of 6 documents.
#
# What actually happened: the model returned a long confident
# response that SOUNDS complete but is missing entire topics:
# - No mention of security policies
# - No mention of financial performance
# - No mention of specific product specifications
# The model never said "I don't have full information" —
# it just answered with what it had, sounding authoritative.
#
# This is dangerous: the response feels complete but isn't.
#
# What would improve this:
# 1. Increase similarity_top_k to 6 to include all documents
# 2. Use a more specific query instead of "everything"
# 3. Add a system prompt telling the model to say
#    "I only have partial information" when context is limited
# 4. Split the query into specific sub-questions per topic

# --- LlamaIndex Q4 ---

# Create Judge LLM
llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# Query 1 — expected high quality
q_good = "What employee benefits does BrightLeaf offer?"
print(f"\n{'='*60}")
print(f"Evaluating: {q_good}")
response_good = query_engine.query(q_good)
print(f"{response_good}")

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q_good, response=response_good)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q_good, response=response_good)
print("Relevancy Result: " + str(relevancy_result.score))

# Query 2 — expected low quality
q_bad = "What is BrightLeaf's policy on cryptocurrency payments?"
print(f"\n{'='*60}")
print(f"Evaluating: {q_bad}")
response_bad = query_engine.query(q_bad)
print(f"{response_bad}")

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q_bad, response=response_bad)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q_bad, response=response_bad)
print("Relevancy Result: " + str(relevancy_result.score))

# What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?
# A score of 1.0 means the response is fully grounded in the
# retrieved chunks — every claim in the answer can be traced
# back to the source documents. The model invented nothing.
# A score of 0.0 indicates the model hallucinated —
# it added information that does not exist in the chunks,
# making the response unreliable regardless of how confident
# it sounds.

# What does a relevancy score measure, and how is it different from faithfulness?
# Relevancy measures whether the response actually addresses
# the user's question. It compares the answer to the query,
# not to the source chunks.
# The difference from faithfulness:
# Faithfulness checks the answer against the chunks
# (did the model invent anything?)
# Relevancy checks the answer against the question
# (did the model actually answer what was asked?)

# Did the scores change between your two queries? If so, why do you think that happened?
# Yes — significantly.
# Good query:  Faithfulness=1.0, Relevancy=1.0
# Bad query:   Faithfulness=0.0, Relevancy=0.0
# Scores changed because the bad query asked about cryptocurrency
# which is not in any BrightLeaf document.
# The model could not find relevant information and failed
# to answer the question — Relevancy=0.0.
# Faithfulness also varied between runs (1.0 and 0.0)
# because LLM-as-a-judge is non-deterministic —
# the judge LLM can interpret the same response differently across runs.

# What is the LLM-as-a-judge approach?
# LLM-as-a-judge uses a powerful language model (gpt-4o-mini)
# to evaluate the quality of another model's response.
# The judge receives the question, the answer, and the source
# chunks, then decides whether the response is faithful
# and relevant.
# It is used instead of simple accuracy metrics because
# RAG responses can be correct but worded differently
# from any reference answer.
# Accuracy compares exact words — "health benefits" would
# not match "medical coverage" even though they mean the same.
# A judge LLM understands meaning, not just word overlap,
# making it far better at assessing true response quality.