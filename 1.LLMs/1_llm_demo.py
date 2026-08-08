from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
	model="gemini-3.6-flash",# its a chatmodel that is tuned over an LLM
	google_api_key=os.getenv("GEMINI_API_KEY"),
    max_completion_token=10
)

response=llm.invoke("Explain the concept of quantum computing in one sentenc e." )

print(response.content[0]["text"])

