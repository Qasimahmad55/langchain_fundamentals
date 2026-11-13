from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}", input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Generate a linkedin post for  {topic}", input_variables=["topic"]
)

parallel_chain = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt1, model, parser),
        "post": RunnableSequence(prompt2, model, parser),
    }
)

print(parallel_chain.invoke({"topic": "Ai"}))
