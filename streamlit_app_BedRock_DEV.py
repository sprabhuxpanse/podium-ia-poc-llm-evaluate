import streamlit as st
import pandas as pd
import boto3

# Define the rubric for scoring
rubric = {
    1: "The model fails to understand the context of user inputs and provides responses that are irrelevant or inappropriate.",
    2: "The model occasionally understands the context but often provides responses that are incomplete or only partially appropriate.",
    3: "The model generally understands the context and provides appropriate responses, though some responses may be lacking in detail or accuracy.",
    4: "The model consistently understands the context and provides suitable and detailed responses, with only occasional minor inaccuracies or omissions.",
    5: "The model excels in understanding the context and consistently provides highly relevant, detailed, and accurate responses."
}

# Map the model name to the correct model ID and Content-Type
model_id_mapping = {
    "Anthropic Claude 1": {"model_id": "anthropic.claude-v1", "content_type": "text/plain"},
    "Anthropic Claude 2": {"model_id": "anthropic.claude-v2", "content_type": "text/plain"},
    "Anthropic Claude 3.0": {"model_id": "anthropic.claude-v3.0", "content_type": "text/plain"},
    "Anthropic Claude 3.5": {"model_id": "anthropic.claude-v3.5", "content_type": "text/plain"},
    "Anthropic Claude 3.5 Instant": {"model_id": "anthropic.claude-v3.5-instant", "content_type": "text/plain"},
    "Cohere Command R 3.0": {"model_id": "cohere.command-r-v3.0", "content_type": "text/plain"},
    "Cohere Command 3.0": {"model_id": "cohere.command-v3.0", "content_type": "text/plain"},
    "Cohere Command Lite": {"model_id": "cohere.command-lite", "content_type": "text/plain"},
    "Meta LLaMA 2 7B Chat": {"model_id": "meta.llama2-7b-chat", "content_type": "text/plain"},
    "Meta LLaMA 2 13B Chat": {"model_id": "meta.llama2-13b-chat", "content_type": "text/plain"},
    "Meta LLaMA 2 70B Chat": {"model_id": "meta.llama2-70b-chat", "content_type": "text/plain"},
    "Amazon Titan Text LLM": {"model_id": "amazon.titan-text-llm", "content_type": "text/plain"},
    "Amazon Titan 3B Chat": {"model_id": "amazon.titan-tg1-large", "content_type": "application/json"},
    "Amazon Titan 1.5B Text": {"model_id": "amazon.titan-tg1-medium", "content_type": "text/plain"},
    "Amazon Titan 1.3B Text": {"model_id": "amazon.titan-tg1-small", "content_type": "text/plain"},
}

def load_csv(file):
    return pd.read_csv(file)

def get_bedrock_client():
    # Initialize the AWS BedRock client
    return boto3.client('bedrock-runtime')

def query_model(client, model_name, input_text):
    model_info = model_id_mapping.get(model_name)
    if model_info is None:
        raise ValueError(f"Invalid model name: {model_name}")

    model_id = model_info["model_id"]
    content_type = model_info["content_type"]

    # Query the BedRock model with the input text
    response = client.invoke_model(
        modelId=model_id,
        body=input_text.encode('utf-8'),
        contentType=content_type
    )
    return response['body'].read().decode('utf-8')

def evaluate_answer(client, evaluator_model, question, candidate_answer):
    # Evaluate the answer using the evaluator model and the provided rubric
    evaluation = query_model(client, evaluator_model, f"Evaluate the following Q&A based on a scale of 1 to 5.\nQuestion: {question}\nAnswer: {candidate_answer}\nEvaluation Criteria: {rubric}")
    score = int(evaluation.strip())  # Assume the response is just a number
    justification = f"The Evaluator gave a score of {score} because {rubric[score].lower()}"
    return score, justification

def main():
    st.title("AWS BedRock Model QA Evaluator")

    uploaded_file = st.file_uploader("Upload a CSV file with a 'Questions' column", type=["csv"])
    
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        if 'Questions' not in df.columns:
            st.error("The uploaded CSV does not contain a 'Questions' column.")
            return
        
        st.write("Uploaded Questions:")
        st.dataframe(df)

        client = get_bedrock_client()
        
        # Select the Candidate and Evaluator models
        candidate_model = st.selectbox("Select Candidate Model", list(model_id_mapping.keys()))
        evaluator_model = st.selectbox("Select Evaluator Model", list(model_id_mapping.keys()))

        if st.button("Generate Answers and Evaluate"):
            answers = []
            ratings = []
            justifications = []

            for question in df['Questions']:
                # Get the answer from the Candidate model
                candidate_answer = query_model(client, candidate_model, question)
                answers.append(candidate_answer)

                # Evaluate the answer using the Evaluator model
                score, justification = evaluate_answer(client, evaluator_model, question, candidate_answer)
                ratings.append(score)
                justifications.append(justification)

            df['Candidate Answer'] = answers
            df['Evaluator Rating'] = ratings
            df['Evaluator Rating Justification'] = justifications

            output_file = '/mnt/data/streamlit_llm_answers.csv'
            df.to_csv(output_file, index=False)

            st.write("Generated Answers and Evaluations:")
            st.dataframe(df)

            st.write("Summary Statistics:")
            st.write(df['Evaluator Rating'].describe())

            st.download_button(
                label="Download the results as CSV",
                data=df.to_csv(index=False),
                file_name="streamlit_llm_answers.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
