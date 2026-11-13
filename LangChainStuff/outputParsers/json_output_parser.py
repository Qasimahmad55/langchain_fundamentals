from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# prompt = template.format()

# result = model.invoke(prompt)

# # print(result.content[0]["text"])

# final_result = parser.parse(result.content[0]["text"])
# print(final_result)

chain = template | model | parser

result = chain.invoke({})

print(result)
