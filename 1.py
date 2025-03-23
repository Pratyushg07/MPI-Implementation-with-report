import openai
import pandas as pd
import pickle
import os
import re
import numpy as np
import sys
from dotenv import load_dotenv

def load_env_variables():
    load_dotenv()
    return os.environ.get("OPENAI_API_KEY")

# Load OpenAI API key
openai.api_key = load_env_variables()

# File path and constants
DATA_FILE = "mpi_120.csv"

def load_dataset(file_path=DATA_FILE):
    df = pd.read_csv(file_path)
    return df

# Define prompt template
PROMPT_TEMPLATE = """Question:
Given a statement about yourself: "You {}."
Please select the most accurate response from the following options:
(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate

Answer:"""

def generate_responses(data, temperature=0):
    results = []
    batch_size = 1 
    
    for start in range(0, len(data), batch_size):
        batch = data.iloc[start : start + batch_size]
        questions = [PROMPT_TEMPLATE.format(row["text"].lower()) for _, row in batch.iterrows()]
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI simulating a human personality. Answer the following questions as if you were a real person."},
                {"role": "user", "content": "\n".join(questions)}
            ],
            temperature=temperature,
            max_tokens=1200,
            top_p=0.95,
        )
        
        for i, choice in enumerate(response.choices):
            results.append((batch.iloc[i], questions[i], choice.message.content))
            print(choice.message.content, batch.iloc[i]["label_ocean"])
    
    with open("results.pkl", "wb") as file:
        pickle.dump(results, file)

# Load dataset and generate responses
dataset = load_dataset(DATA_FILE)
generate_responses(dataset)

# Load saved responses
with open("results.pkl", "rb") as file:
    responses_data = pickle.load(file)

# Initialize scoring variables
response_counts = {key: 0 for key in ['A', 'B', 'C', 'D', 'E', 'UNK']}
personality_scores = {trait: [] for trait in ["O", "C", "E", "A", "N"]}
score_mapping = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}

def compute_statistics(data):
    mean_values = {key: np.mean(values) for key, values in data.items()}
    std_values = {key: np.std(values) for key, values in data.items()}
    
    return f"""Mean Scores:
{sorted(mean_values.items())}\n
Standard Deviations:
{sorted(std_values.items())}"""

# Process responses
for entry in responses_data:
    response_text = entry[2]
    match = re.search(r'\((A|B|C|D|E)\)', response_text, re.IGNORECASE)
    
    if match:
        choice = match.group(1).upper()
        response_counts[choice] += 1
        trait_label = entry[0]['label_ocean']
        response_key = entry[0]['key']
        score = score_mapping[choice]
        
        if response_key == 1:
            personality_scores[trait_label].append(score)
        else:
            personality_scores[trait_label].append(6 - score)

# Print results
print(compute_statistics(personality_scores))
print(response_counts)
