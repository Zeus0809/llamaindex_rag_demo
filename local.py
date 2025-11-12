from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llamaindex_utils.integrations import LlamaCppEmbedding, DockerLLM

EMBED_MODEL_PATH = "/Users/illiakozlov/Desktop/Projects/ChatWithPDF/chat-with-pdf/local_models/embed/nomic-embed-text-v2-moe.Q8_0.gguf"

# Initialize our own embedding model (taken from HuggingFace)
Settings.embed_model = LlamaCppEmbedding(model_path=EMBED_MODEL_PATH, verbose=False)

# Initialize our own chat model (running from Docker)
our_local_llm = DockerLLM(model="ai/gemma3n")

documents = SimpleDirectoryReader("data").load_data()
print("Document loaded!")
index = VectorStoreIndex.from_documents(documents)
print("Index created!")
query_engine = index.as_query_engine(llm=our_local_llm)
print("Query engine created!")
response = query_engine.query("Who is Illia?")
print("\nLLM response: ", response)
