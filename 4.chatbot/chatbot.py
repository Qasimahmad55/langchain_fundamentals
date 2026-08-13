from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

chat_history = [
    SystemMessage(content="You are a helpful assistant")
]  # for storing the chat messages

while True:
    user_Input = input("You: ")
    chat_history.append(HumanMessage(content=user_Input))
    
    if user_Input == "exit":
        break
    result = llm.invoke(chat_history)
    
    chat_history.append(AIMessage(content=result.content[0]["text"]))
    
    print("AI: ", result.content[0]["text"])

print(chat_history)
