from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os

model = ChatGoogleGenerativeAI(
	model="gemini-3.6-flash",# its a chatmodel that is tuned over an LLM
	google_api_key=os.getenv("GEMINI_API_KEY"),
    # max_completion_tokens=10
)

st.header("Research Tool")

user_input = st.text_input("Enter your prompt")

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.write(result.content[0]['text'])
