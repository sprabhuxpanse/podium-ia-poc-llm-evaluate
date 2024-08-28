# DOCUMENTATION: using mlflow.log_table (https://mlflow.org/docs/latest/python_api/mlflow.html?highlight=log_table#mlflow.log_table)

import pprint
import boto3
import pandas as pd
from langchain_community.embeddings import BedrockEmbeddings
#from langchain_community.llms import Bedrock # OLD BedRock
from langchain.llms.bedrock import Bedrock # NEW BedRock
from langchain_community.chat_models import BedrockChat
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain.llms import OpenAI
import os
import mlflow
import logging
logging.getLogger("mlflow").setLevel(logging.DEBUG)

# AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
#from langchain_core.prompts import PromptTemplate 

# for GPT 3.5 or 4o
#os.environ["AZURE_OPENAI_API_KEY"] = "4ca0935cd5fd496b9e5d83fc9b6abe2d" # gpt-3.5 /4o
#os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openaifreedomaiprod.openai.azure.com/" # gpt-3.5/4o

# GPT-4o
#os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01" # GPT-4o
#os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4o" # GPT-4o
#model_id ="gpt-4o"

# GPT-3.5
#os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01" # GPT-3.5, Documentation # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt?tabs=python-new
#os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-35-turbo" # GPT-3.5
#model_id = "gpt-35-turbo"

# GPT-4
os.environ["AZURE_OPENAI_API_KEY"] = "023a81ce74c343cfa7921d1d51f71457"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openaifreedomaisandbox-useast2.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-05-01-preview" # turbo-2024-04-09 
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4" # GPT-4
model_id = "gpt-4"


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow.dev.fhmc.xpanse-ai.com:8443"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'

mlflow.set_experiment("langchain-rag-bedrock-callaid_dh_05_23_2024")

#Define vectorstore
global vectorstore_faiss

def config_llm_azure():

    llm = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        #temperature=0.7
    )

    return llm


def config_llm_huggingface():
    client = boto3.client('iam')

    repo_id = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"

    llm = HuggingFaceHub(repo_id=repo_id, task="text-generation", huggingfacehub_api_token="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")
    #llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")
    
    #llm_chain = LLMChain(prompt=prompt_template, llm=llm)

    return llm, repo_id

def config_vector_db(filename):
    client = boto3.client('bedrock-runtime')
    bedrock_embeddings = BedrockEmbeddings(client=client)
    #loader = Docx2txtLoader(filename)
    loader = DirectoryLoader(filename) # UPDATE OR REMOVE
    pages = loader.load_and_split()
    vectorstore_faiss = FAISS.from_documents(pages, bedrock_embeddings)
    return vectorstore_faiss

def vector_search (query):
    docs = vectorstore_faiss.similarity_search_with_score(query)
    info = ""
    for doc in docs:
        info+= doc[0].page_content+'\n'
    return info    

llm = config_llm_azure()
vectorstore_faiss = config_vector_db("call_aids_data") 

# Brian V2 Prompt
prompt_version = "promptV2"
my_template = """


You are a customer service agent in the mortgage industry. 
 
You specialize in escrow related questions and making sure customers understand changes to their home loans.  
 
Your customers have questions that involve referencing regulatory information  
Your customers often have no background in the mortgage industry or its rules and regulations, and require detailed answers that will clarify the policy information and ensure they understand how to stay in compliance. It's also critical to mention specific resources if applicable so that the customer understands the policy references and feels confident about the answer.  
 
The response should also note regulatory references when creating the regulations in reference  
When answering the question - make sure to sound friendly, but be clear and direct in explaining the answer.  
when justifying the answer in detail, make sure to say "in compliance with the regulation name and code number where applicable - present the information from a regulatory focused perspective as much as possible - where possible, provide insight into the policies and regulations so that the customer has as much context and feels confident of the answer bring provided  
Close each response with "Are you satisfied with the answer to the question? If not tell me how I can improve it." 

Human:

    <Information>
    {info}
    </Information>

    {input}

Assistant:

"""

prompt_template = PromptTemplate(input_variables=['input', 'info'], template=my_template)
question_chain = LLMChain(llm = llm, prompt = prompt_template, output_key = "answer")

# https://github.com/Xpanse-AI/sagemaker-llm-evaluations/blob/main/test_dataset/mistral_7b_finetuning_qa_dataset.csv
questions =             {"inputs":
                         ["What dictates the delivery of the short-year statement to the customer?",
                          "What happens if a loan is more than 30 days past due at the time of escrow account analysis?",
                          "How can a new Servicer determine the new escrow account computation year upon the transfer of servicing?",
                          "What are the requirements for a customer to be eligible for escrow removal?",
                          "How long must a customer maintain an escrow account for HPML loans?",
                          "What happens if a customer wants to spread their escrow shortage payment over more than 12 months?",
                          "What are the insurance policy types required at origination and throughout the life of a Mortgage loan?",
                          "What is Lender-Placed Insurance?",
                          "How can customers pay for Private Mortgage Insurance premiums?",
                          "How can an FHA loan benefit a borrower?",
                          "How long does a customer have to cash a refund or escrow check before it becomes void?",
                          "What action is taken if the escrow refund check is not cashed after 60 days from issue?"
                         ]}

questions_list_of_lists = [question for question in questions.values()]
questions_list = [x for xs in questions_list_of_lists for x in xs]


#table_dict = {}
table_dict = dict.fromkeys(["inputs", "outputs"], [])
answers_list = []

for question in questions_list:
    info = vector_search(question)
    answer = question_chain.run({'input' : question, 'info':info})

    answers_list.append(answer)

    print(f"QUESTION: {question}\n")
    print(f"ANSWER: {answer}\n")

table_dict = {"inputs": questions_list, "outputs": answers_list}

pprint.pprint(table_dict)

#df = pd.DataFrame.from_dict(table_dict)

azure = "azure"

with mlflow.start_run(run_name=f"{azure}-{model_id}-{prompt_version}"):
    model_info = mlflow.langchain.log_model(question_chain, "langchain_model")
    #mlflow.log_table(data=df, artifact_file="qabot_eval_results.json")
    mlflow.log_table(data=table_dict, artifact_file="qabot_eval_results.json")  
    print("Results logged to MLFlow")