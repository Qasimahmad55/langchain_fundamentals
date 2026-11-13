from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from schema import AnswerQuestion
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.messages import HumanMessage

load_dotenv()
pydanticParser = PydanticToolsParser(tools=[AnswerQuestion])
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# Actor Agent Prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI researcher.
Current time: {time}
1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

first_reponder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

first_reponder_chain = (
    first_reponder_prompt_template
    | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
    | pydanticParser
)

reponse = first_reponder_chain.invoke(
    {
        "messages": [
            HumanMessage(
                content="write a short blog post on how small bussiness can leverage Ai to grow"
            )
        ]
    }
)

print(reponse)

