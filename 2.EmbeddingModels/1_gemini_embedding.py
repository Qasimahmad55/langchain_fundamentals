from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vectors=embeddings.embed_query("what is embedding",output_dimensionality=32)

print (vectors)


