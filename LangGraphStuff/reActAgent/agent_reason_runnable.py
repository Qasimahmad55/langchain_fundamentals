"""The Gemini model and the tools it can use."""

from datetime import datetime
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

llm = ChatGroq(model="openai/gpt-oss-20b")


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    search = TavilySearch(search_depth="basic", max_results=3)
    return str(search.invoke({"query": query}))


tools = [get_current_time, web_search]

# Give the model the tools so it can choose one.
llm_with_tools = llm.bind_tools(tools)
