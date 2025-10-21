from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent


# configuring components of rag
model = ChatOllama(model="qwen3:4b", temperature=0)
embeddings = OllamaEmbeddings(model="embeddinggemma")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="constitution_docs",
    connection="postgresql+psycopg://postgres:777adlet@localhost:5432/vectors",
)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
    )
    return serialized, retrieved_docs


agent = create_agent(
    model=model,
    tools=[retrieve_context],
    system_prompt="You have access to a tool that retrieves context from a PDF File. Use the tool to help answer user queries. Asnwer in Russian language",
)

# loading docs
loader = PyPDFLoader("file.pdf")

docs = loader.load()

# splitting documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(docs)

# storing documents
document_ids = vector_store.add_documents(documents=all_splits)

query = ("Кто возглавляет палаты?")

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()