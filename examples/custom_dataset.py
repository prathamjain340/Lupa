# examples/custom_dataset.py
#
# Demonstrates generate_dataset() — create a Lupa evaluation dataset
# from your own documents using an LLM.
#
# Requires OPENAI_API_KEY environment variable.
#
# Run: python examples/custom_dataset.py

import json

import rageval

# ------------------------------------------------------------------
# Source documents — short finance paragraphs used as the raw material.
# In real usage these would come from your own corpus (PDFs, databases, etc.)
# ------------------------------------------------------------------

DOCUMENTS = [
    (
        "Tesla reported Q2 2024 revenue of $25.5 billion, up 2% year over year. "
        "Automotive revenue was $19.9 billion, while Energy Generation and Storage "
        "revenue reached a record $3.0 billion, an increase of 100% year over year. "
        "The company delivered 443,956 vehicles during the quarter."
    ),
    (
        "Nvidia posted fiscal Q1 2025 revenue of $26.0 billion, up 262% year over year, "
        "driven by Data Center revenue of $22.6 billion. Gross margin expanded to 78.4%. "
        "The company guided Q2 2025 revenue of approximately $28.0 billion."
    ),
    (
        "JPMorgan Chase reported Q1 2024 net income of $13.4 billion, or $4.44 per share, "
        "on revenue of $41.9 billion. Net interest income rose 11% to $23.2 billion. "
        "The firm's CET1 capital ratio stood at 15.0% at quarter end."
    ),
    (
        "Amazon's AWS segment generated Q1 2024 revenue of $25.0 billion, up 17% year "
        "over year, with operating income of $9.4 billion. Total company revenue was "
        "$143.3 billion for the quarter, a 13% increase from Q1 2023."
    ),
]

# ------------------------------------------------------------------
# Generate dataset — 3 questions, saved to custom_dataset.json
# ------------------------------------------------------------------

print("Generating dataset from 4 documents (3 questions)...")
print("This calls the OpenAI API — requires OPENAI_API_KEY in your environment.\n")

dataset = rageval.generate_dataset(
    documents=DOCUMENTS,
    n_questions=3,
    llm="openai",
    output_path="custom_dataset.json",
)

print(f"Generated {len(dataset)} examples. Saved to custom_dataset.json.\n")

# ------------------------------------------------------------------
# Show the first example so the output format is visible
# ------------------------------------------------------------------

print("First generated example:")
print(json.dumps(dataset[0], indent=2))
