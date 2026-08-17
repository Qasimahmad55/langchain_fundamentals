from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")


# schema
class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[
        Literal["positive,negative"], "Return a sentiment of the review"
    ]
    pros: Annotated[Optional[list[str]], "Write down all of the pros inside the list"]
    cons: Annotated[Optional[list[str]], "Write down all of the cons inside the list"]


structured_model = llm.with_structured_output(Review)

result = structured_model.invoke(
    "I ordered this laptop after reading several positive reviews, and after using it for nearly a month, I have to say my experience has been a bit complicated. The overall build quality is excellent, the keyboard feels great, and performance is more than enough for my daily development work, including running Docker containers and multiple browser tabs at the same time. The display is also sharp and bright, although the colors look slightly washed out compared with my older laptop. My biggest complaint is the battery life. Under light usage it lasts reasonably well, but once I start compiling projects or running heavier applications, the battery drops surprisingly quickly. I also noticed that the fans become quite loud under sustained workloads, which isn't necessarily a dealbreaker but can be distracting. On the positive side, the laptop stays responsive even when I'm pushing it hard, and I haven't experienced any crashes or serious software problems so far. The charger is another minor annoyance because it's noticeably larger and heavier than I expected. Considering the price, I was initially expecting fewer compromises, but after using it for several weeks, I still think it's a solid machine for developers who prioritize performance and build quality over portability and battery life. I wouldn't call it perfect, but I don't regret buying it either."
)

print(result)
