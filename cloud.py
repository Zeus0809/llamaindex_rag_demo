from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

documents = SimpleDirectoryReader("data").load_data()
print("Document loaded!")
index = VectorStoreIndex.from_documents(documents)
print("Index created!")
query_engine = index.as_query_engine()
print("Query engine created!")
response = query_engine.query("Who is Illia?")
print("\nLLM response: ", response)

