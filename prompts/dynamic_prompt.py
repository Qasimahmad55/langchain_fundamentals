import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

st.header("Dynamic Prompting")
st.write(
    "This template takes inputs from the Streamlit UI and formats the prompt dynamically."
)

user_topic = st.text_input("Enter a tech topic:", value="Docker")
user_tone = st.selectbox(
    "Choose a tone:", ["Professional", "Funny", "Explain like I'm 5"]
)

if st.button("Run dynamic prompt"):
    with st.spinner("Generating"):
        prompt_template = PromptTemplate(
            template="Explain {topic} in 2 concise sentences using a {tone} tone.",
            input_variables=["topic", "tone"],
        )

    formatted_prompt = prompt_template.format(topic=user_topic, tone=user_tone)
    response = llm.invoke(formatted_prompt)
    
    st.success(response.content[0]["text"])
