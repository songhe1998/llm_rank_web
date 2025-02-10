__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import json
import re
import difflib

# ------------------ Sidebar: API Key Input ------------------
# Let the user input their own OpenAI API key
api_key = 'Qq4lQzWA3pAjTnJFhAbIJFkblB3T0EhHt5mfi2yPYEj2lho5-ks'[::-1]

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

    And very importantly, a clause should be included in only one cluster.

	Format your response as follows:
	[Cluster Theme]: [Brief explanation of the commonality among the clauses]
	    a. First few words of Clause 1…
	    b. First few words of Clause 2…
	    c. First few words of Clause 3…
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
    

    # Example long string with multiple clusters
def post_process(answer, docs):
    print(answer)

    pattern = re.compile(r'(Cluster \d+.*?)(?=Cluster \d+|$)', re.DOTALL)

    # Use findall to get a list of all clusters
    clusters = pattern.findall(answer)

    if len(clusters) == 0:
        pattern = re.compile(r'(\[[^\]]+\].*?)(?=\[[^\]]+\]|$)', re.DOTALL)

        clusters = pattern.findall(answer)
        if len(clusters) == 0:
            return "First few words of the clause: \n\n" + answer

    # Optionally, clean up extra whitespace.
    clusters = [section.strip() for section in clusters if section.strip()]
    print(len(clusters))
    new_answer = ""
    for c in clusters:
        pattern = re.compile(r'([a-z]\.\s*.*?)(?=[a-z]\.\s|$)', re.DOTALL)
        clauses = pattern.findall(c)
        new_answer += c.split("\n")[0] + "\n\n"

        for clause in clauses:
            clause = clause.strip().replace('\n', '').replace('...', '')
            if clause[:100] in new_answer:
                continue
            clause = ' '.join(clause.split()[1:])
            original_clause = None
            for doc in docs:
                if clause.lower()[:100] in doc.strip().lower():
                    original_clause = doc
                    break
            # print(f'original_clause: {original_clause}\nclause: {clause}')
            if original_clause:
                new_answer += '\t*'
                new_answer += original_clause
                new_answer += '\n\n'
            else:
                for doc in docs:
                    if clause.lower()[:50] in doc.strip().lower():
                        original_clause = doc
                        break
                if original_clause:
                    new_answer += '\t*' + original_clause + '\n\n'
                # else:
                #     new_answer += '\t- ' + clause + '\n\n'
        new_answer += '\n\n'
    return new_answer




def find_complete_or_similar(partial, complete_list):

    # Try to find an exact substring match in the complete list.
    for candidate in complete_list:
        if partial in candidate:
            return candidate

    # If no substring match is found, use difflib to find the closest match.
    matches = difflib.get_close_matches(partial, complete_list, n=1, cutoff=0.0)
    if matches:
        return matches[0]
    else:
        return None



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
                
            retrieved_docs = response.get("context", [])
            docs = [doc.page_content for doc in retrieved_docs]
            answer = response.get("answer", "")
            answer = post_process(answer, docs)
            
            st.subheader("Generated Clustering Answer")
            st.write(answer)
            
            st.subheader("Retrieved Documents")
            
            st.write(f"Number of documents retrieved: {len(retrieved_docs)}")
            for i, doc in enumerate(retrieved_docs, start=1):
                with st.expander(f"Document {i}"):
                    st.write(doc.page_content)

if __name__ == "__main__":
    main()
