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

def load_csv(file):
    return pd.read_csv(file)

def get_bedrock_client():
    # Initialize the AWS BedRock client
    return boto3.client('bedrock-runtime')

def query_model(client, model_id, input_text):
    # Query the BedRock model with the input text
    response = client.invoke_model(
        modelId=model_id,
        body=input_text.encode('utf-8'),
        contentType='text/plain'
    )
    return response['body'].read().decode('utf-8')

def evaluate_answer(evaluator_model, question, candidate_answer):
    # Evaluate the answer using the evaluator model and the provided rubric
    evaluation = query_model(evaluator_model, f"Evaluate the following Q&A based on a scale of 1 to 5.\nQuestion: {question}\nAnswer: {candidate_answer}\nEvaluation Criteria: {rubric}")
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
        candidate_model = st.selectbox("Select Candidate Model", ["anthropic.claude-v2:1", "amazon.titan-text-express-v1"])
        evaluator_model = st.selectbox("Select Evaluator Model", ["anthropic.claude-v2:1", "amazon.titan-text-express-v1"])


        if st.button("Generate Answers and Evaluate"):
            answers = []
            ratings = []
            justifications = []

            for question in df['Questions']:
                # Get the answer from the Candidate model
                candidate_answer = query_model(client, candidate_model, question)
                answers.append(candidate_answer)

                # Evaluate the answer using the Evaluator model
                score, justification = evaluate_answer(client, question, candidate_answer)
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
