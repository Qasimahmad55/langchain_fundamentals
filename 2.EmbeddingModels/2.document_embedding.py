from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vectors=embeddings.embed_documents([
    "Hey there whats up",
    "hi how are you",
    "what is the capital of pakistan"
],output_dimensionality=32)

print (vectors)


