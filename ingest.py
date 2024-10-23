from llama_index.core import SimpleDirectoryReader

# Load db data (Kenya National Medicines Formulary)
def ingest():
    pdf_path = "data"
    documents = SimpleDirectoryReader(pdf_path).load_data()
    print('KNMF Loaded')
    return documents
