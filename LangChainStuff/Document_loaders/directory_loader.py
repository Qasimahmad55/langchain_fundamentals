from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(path="Document_loaders", glob="*.pdf", loader_cls=PyPDFLoader)

# docs = loader.load() # used for instantly loading documents

docs = loader.lazy_load()  # used for handling a large amount of documents

print(docs[0].page_content)
