from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
model2 = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt1 = PromptTemplate(
    template="Generate a short and simple notes from the following text \n {text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n  notes-> {notes} and quiz->{quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser,
    }
)

merge_chain = prompt3 | model1 | parser

final_chain = parallel_chain | merge_chain

text = """Artificial Intelligence (AI) is a technology that allows computers to perform tasks that normally require human intelligence. These tasks include understanding language, recognizing images, solving problems, and making decisions. AI systems learn from data and use patterns in that data to produce useful results.Machine Learning (ML) is a branch of Artificial Intelligence. Instead of programming every rule manually, developers provide a machine learning model with data so that it can learn patterns. There are three common types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.Supervised learning uses labeled data to train a model. For example, a model can be trained using pictures labeled as cats or dogs and then learn to classify new pictures. Unsupervised learning works with unlabeled data and tries to discover hidden patterns or groups. Reinforcement learning allows an agent to learn by interacting with an environment and receiving rewards or penalties for its actions.Deep Learning is another important area of AI. It uses neural networks with many layers to learn complex patterns from large amounts of data. Deep learning is commonly used in image recognition, speech recognition, natural language processing, and recommendation systems.
Natural Language Processing (NLP) focuses on enabling computers to understand and process human language. NLP is used in chatbots, translation systems, sentiment analysis, text summarization, and question-answering systems.
AI has many practical applications. It is used in healthcare to help analyze medical images, in banking to detect fraudulent transactions, in transportation to improve navigation, and in education to create personalized learning experiences. However, AI also creates challenges related to privacy, security, bias, and responsible use of technology.
"""

result = final_chain.invoke({"text": text})

print(result)
