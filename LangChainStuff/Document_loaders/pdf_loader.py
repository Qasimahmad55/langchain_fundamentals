from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt = PromptTemplate(
    template="write a summary for the following pdf {pdf}", input_variables=["pdf"]
)

loader = PyPDFLoader("Document_loaders/Software_Engineer-1.pdf")

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({"pdf": docs[0].page_content}))
