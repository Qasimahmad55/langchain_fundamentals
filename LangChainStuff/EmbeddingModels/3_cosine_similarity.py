from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

documents = [
    "Python is a popular programming language used for data science and AI.",
    "The recipe requires flour, eggs, milk, and sugar.",
    "Machine learning models rely on linear algebra and calculus.",
    "Football is played on a rectangular field with a goal at each end.",
] 

query = "What ingredients do I need to make a basic cake batter?"

query_vector = embeddings.embed_query(query)

doc_vector = embeddings.embed_documents(documents)

similarity_matirx = cosine_similarity([query_vector], doc_vector)

scores = similarity_matirx[0]

results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

print(f"\n Query: '{query}'\n")
print("Search Results (Ranked by Scikit-Learn Cosine Similarity):")
for doc, score in results:
    print(f"[{score:.4f}] {doc}")
