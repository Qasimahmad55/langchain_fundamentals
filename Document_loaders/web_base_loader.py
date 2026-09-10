import os

os.environ.setdefault("USER_AGENT", "langchain-models-web-loader/1.0")

from langchain_community.document_loaders import WebBaseLoader

url = "https://blog.hubspot.com/website/html-websites#understanding-the-basics-of-html-websites"

loader = WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content)
