from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

FILE = Path(__file__).parent / "nodejs.pdf"

load_dotenv()

loader = PyPDFLoader(file_path=FILE)

docs = loader.load()

chunks_generator = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=350)

split_docs = chunks_generator.split_documents(docs)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="nodejs",
    embedding=embedding_model,
    force_disable_check_same_thread=True,
)

print("Doc indexing complete!")
