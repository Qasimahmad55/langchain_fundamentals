from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(path="Document_loaders", glob="*.pdf", loader_cls=PyPDFLoader)

docs = loader.load()

print(docs[0].page_content)
