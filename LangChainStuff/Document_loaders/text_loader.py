from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
parser = StrOutputParser()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt = PromptTemplate(
    template="write a summary for the following poem {poem}", input_variables=["poem"]
)

loader = TextLoader("Document_loaders/cricket.txt")

docs = loader.load()

# print(type(docs))
# print(len(docs))

chain = prompt | model | parser

print(chain.invoke({"poem": docs[0].page_content}))
