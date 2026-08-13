from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

chat_history = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about langchain"),
]

result = llm.invoke(chat_history)

chat_history.append(AIMessage(content=result.content[0]["text"]))

print(chat_history)
