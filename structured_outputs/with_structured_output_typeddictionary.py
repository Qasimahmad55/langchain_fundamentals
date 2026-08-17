from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")


# schema
class Review(TypedDict):
    summary: str
    sentiment: str


structured_model = llm.with_structured_output(Review)

result = structured_model.invoke(
    "The phone looks nice, but unfortunately the battery drains incredibly fast. I have to charge it twice a day even with normal usage. The camera is also disappointing in low-light conditions."
)

print(result)
