__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import json

# ------------------ Sidebar: API Key Input ------------------
# Let the user input their own OpenAI API key
api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password",
    help="Your OpenAI API key will be used to authenticate requests."
)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.sidebar.warning("Please enter your OpenAI API key to run the app.")

# ------------------ Import and Initialization ------------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain.schema import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Initialize LLM, embeddings, and vector store.
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./ledgar_embed"  # Directory for persisting the vector store
)

# Load and customize the prompt from LangChain Hub.
prompt = hub.pull("rlm/rag-prompt")
prompt.messages[0].prompt.template = """
	Analyze the following retrieved clauses and categorize them into distinct clusters based on their meaning and relevance to the provided user input clause. Your clustering should be based on your understanding of the clauses’ content and their potential connection to the input.

	For each cluster, provide a clear and concise summary of its theme. Then, list all the clauses under their respective cluster, but only include the first few words of each clause to represent them while ensuring that all retrieved clauses are listed.

    Within each cluster, rank the clauses from most relevant to least relevant based on their alignment with the user input clause.

	Format your response as follows:
		•	[Cluster Theme]: [Brief explanation of the commonality among the clauses]
	•	First few words of Clause 1…
	•	First few words of Clause 2…
	•	First few words of Clause 3…
…

	Ensure that every retrieved clause is included in your clustering. If a clause does not fit into an existing cluster, create a new one with an appropriate theme. Do not summarize or modify the clauses—only list their beginning few words followed by ellipses.
User input clause: {question}
Retrieved clauses: {context} 
Answer:
"""

# ------------------ Application State and Functions ------------------

# Define the application state as a TypedDict
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Step 1: Retrieve documents based on similarity
def retrieve(state: State, k=50):
    retrieved_docs = vector_store.similarity_search(state["question"], k=k)
    return {"context": retrieved_docs}

# Step 2: Generate clustering answer using the prompt and LLM
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Validate if text is non-empty
def is_valid_text(text):
    return isinstance(text, str) and text.strip() != ""

# Main RAG function that optionally adds texts to the vector store before running the process.
def rag(question, texts=None, k=50):
    if texts is not None:
        documents = [Document(page_content=text) for text in texts]
        _ = vector_store.add_documents(documents=documents)
    
    # Wrap retrieval with the chosen 'k' value.
    def retrieve_with_k(state: State):
        return retrieve(state, k=k)
    
    graph_builder = StateGraph(State).add_sequence([retrieve_with_k, generate])
    graph_builder.add_edge(START, "retrieve_with_k")
    graph = graph_builder.compile()
    response = graph.invoke({"question": question})
    return response

# ------------------ Streamlit UI ------------------

def main():
    st.title("Legal Clause Clustering with RAG")
    st.write(
        "This web app uses a Retrieval-Augmented Generation (RAG) approach to analyze a legal clause, "
        "retrieve similar clauses from a document store, and cluster them by meaning and relevance."
    )
    
    st.sidebar.header("Options")
    
    # Optional file uploader for legal clauses (JSONL format expected).
    uploaded_file = st.sidebar.file_uploader("Upload JSONL file with clauses (optional)", type=["jsonl"])
    texts = None
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8").splitlines()
            texts = []
            for line in content:
                try:
                    data = json.loads(line)
                    if "provision" in data:
                        text = data["provision"]
                        if is_valid_text(text):
                            texts.append(text)
                except Exception as e:
                    st.error(f"Error parsing line: {e}")
            texts = texts[:5000]  # Limit to the first 5000 clauses
            st.sidebar.success(f"Loaded {len(texts)} clauses from file.")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")

    # Input for the legal clause to analyze
    default_clause = (
        'For the purposes of this Agreement, “Confidential Information” means any non-public, proprietary, '
        'or sensitive information disclosed by one party (“Disclosing Party”) to the other party (“Receiving Party”), '
        'whether in written, oral, electronic, or other tangible or intangible form, including but not limited to '
        'business plans, financial data, customer lists, trade secrets, and proprietary technology.'
    )
    user_clause = st.text_area("Enter a legal clause to analyze:", value=default_clause, height=150)
    
    # Slider for choosing the number of documents to retrieve
    k = st.sidebar.slider("Number of documents to retrieve", min_value=10, max_value=100, value=50, step=5)
    
    if st.button("Run Analysis"):
        if not api_key:
            st.error("Please enter your API key in the sidebar.")
            return
        if not is_valid_text(user_clause):
            st.error("Please enter a valid legal clause.")
        else:
            with st.spinner("Processing..."):
                response = rag(user_clause, texts=None, k=k)
            
            st.subheader("Generated Clustering Answer")
            st.write(response["answer"])
            
            st.subheader("Retrieved Documents")
            retrieved_docs = response.get("context", [])
            st.write(f"Number of documents retrieved: {len(retrieved_docs)}")
            for i, doc in enumerate(retrieved_docs, start=1):
                with st.expander(f"Document {i}"):
                    st.write(doc.page_content)

if __name__ == "__main__":
    main()
