import streamlit as st
from podium_ia_poc_streamlit.podium_streamlit import PodiumAIStreamlit
podium_streamlit = PodiumAIStreamlit()

APP_NAME = "My Streamlit App"
PAGE_TITLE = "My Streamlit App Page Title"

def on_llm_model_select_change():
    print(f"{st.session_state['username']} | Selected LLM model: {st.session_state['llm_model']}")

@podium_streamlit.setup(app_name=APP_NAME, page_title=PAGE_TITLE, env="prod")
def main():

    # TODO:
    #### Your Streamlit app code goes here ...
    supported_models = (
        "GPT-4o",
        "mistral.mistral-large-2402-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
    )
    
    st.sidebar.selectbox(
        "Which model would you like to use?",
        supported_models,
        on_change=on_llm_model_select_change,
        key="llm_model",
    )

    st.write("Welcome to my Streamlit app!")
    st.write(f"Selected LLM model: {st.session_state['llm_model']}")

    # To get a variable from environment or aws secrets manager, use the following
    value = podium_streamlit.get_variable("BEDROCK_AWS_REGION")
    st.write(f"AWS Bedrock region: {value}")

main()