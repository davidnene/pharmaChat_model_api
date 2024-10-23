from llama_index.core import VectorStoreIndex, Settings
from ingest import ingest
from model_wrapper import llm

def index():
    documents = ingest()
    # Load vector db
    index = VectorStoreIndex.from_documents(documents)

    Settings.llm = llm
    Settings.chunk_size = 1024
    
    return index