from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")


# schema
Review = {
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Write down all the key themes discussed in the list of reviews",
        },
        "summary": {"type": "string", "description": "A brief summary of the review"},
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative"],
            "description": "Return a sentiment of the review",
        },
        "pros": {
            "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
            "default": None,
            "description": "Write down all of the pros inside the list",
        },
        "cons": {
            "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
            "default": None,
            "description": "Write down all of the cons inside the list",
        },
    },
    "required": ["key_themes", "summary", "sentiment"],
    "additionalProperties": False,
}


structured_model = llm.with_structured_output(Review)

result = structured_model.invoke(
    "I've been using this laptop for about three months. Performance has been excellent and it handles development work without any problems. The keyboard is comfortable too, although the speakers are weaker than I expected."
)

print(result)
