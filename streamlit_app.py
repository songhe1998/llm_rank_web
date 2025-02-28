# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import json
import re
import difflib
import numpy as np  # NEW: For computing cosine similarities

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

# ------------------ New Prompt for Topic-Based Clustering ------------------
topic_prompt_template = """
Analyze the following retrieved clauses and categorize them into distinct clusters based on the provided topics. The topics to group by are: {topics}.

For each topic, list all the clauses that relate to it, only including the first few words of each clause to represent them. If a clause does not clearly belong to any of the provided topics, include it in a separate cluster labeled [Other].

Format your response as follows:
[Topic Name]: [Brief explanation of how the clauses relate to the topic]
    a. First few words of Clause 1...
    b. First few words of Clause 2...
    c. First few words of Clause 3...
    
[Other]: [Explanation for clauses not fitting any topic]
    a. First few words of Clause 1...
    b. First few words of Clause 2...

Ensure that every retrieved clause is included in one of the clusters.
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
def retrieve(question: str, k=50):
    return vector_store.similarity_search(question, k=k)

# New: Generate topic-based clustering using provided topics
def generate_topics(topics: str, docs: list):
    docs_content = "\n\n".join(docs)
    prompt_message = topic_prompt_template.format(topics=topics, context=docs_content)
    messages = [{"role": "user", "content": prompt_message}]
    response = llm.invoke(messages)
    return response.content

# Validate if text is non-empty
def is_valid_text(text):
    return isinstance(text, str) and text.strip() != ""

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


# ------------------ Post-Processing Function ------------------
# def post_process(answer, docs):
#     """
#     Process the LLM output and ensure that every retrieved clause (from docs)
#     is included in one of the clusters. For any clause not originally clustered,
#     compute its embedding and assign it to the closest cluster.
#     """
#     st.write("LLM output:", answer)  # For debugging/logging

#     # Try multiple regex patterns to extract clusters
#     cluster_patterns = [
#         r'(Cluster \d+.*?)(?=Cluster \d+|$)', 
#         r'(\[[^\]]+\].*?)(?=\[[^\]]+\]|$)'
#     ]
#     clusters_raw = None
#     for pat in cluster_patterns:
#         pattern = re.compile(pat, re.DOTALL)
#         clusters_raw = pattern.findall(answer)
#         if clusters_raw:
#             break
#     if not clusters_raw:
#         return "First few words of the clause: \n\n" + answer

#     clusters_raw = [section.strip() for section in clusters_raw if section.strip()]

#     # Build a structured list of clusters.
#     clusters = []
#     for raw in clusters_raw:
#         lines = raw.splitlines()
#         if not lines:
#             continue
#         header = lines[0].strip()
#         # Extract bullet items (e.g., "a. ...", "b. ...", etc.)
#         clause_pattern = re.compile(r'([a-z]\.\s*.*?)(?=[a-z]\.\s|$)', re.DOTALL)
#         bullet_snippets = clause_pattern.findall(raw)
#         bullet_snippets = [snippet.strip().replace('...', '') for snippet in bullet_snippets if snippet.strip()]
#         clusters.append({
#             'header': header,
#             'snippets': bullet_snippets,
#             'docs': []
#         })

#     # Keep track of which docs have been matched to a cluster.
#     matched_doc_indices = set()

#     for cluster in clusters:
#         for snippet in cluster['snippets']:
#             snippet_clean = ' '.join(snippet.split()[1:]).strip()
#             found_doc = None
#             for idx, doc in enumerate(docs):
#                 if idx in matched_doc_indices:
#                     continue
#                 if snippet_clean.lower() in doc.lower():
#                     found_doc = doc
#                     matched_doc_indices.add(idx)
#                     break
#             if found_doc:
#                 cluster['docs'].append(found_doc)

#     cluster_embeddings = []
#     for cluster in clusters:
#         if cluster['docs']:
#             vectors = [np.array(embeddings.embed_query(doc)) for doc in cluster['docs']]
#             avg_vector = np.mean(vectors, axis=0)
#         else:
#             avg_vector = np.array(embeddings.embed_query(cluster['header']))
#         cluster_embeddings.append(avg_vector)

#     def cosine_similarity(vec_a, vec_b):
#         return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8)

#     for idx, doc in enumerate(docs):
#         if idx in matched_doc_indices:
#             continue
#         doc_vector = np.array(embeddings.embed_query(doc))
#         best_similarity = -1
#         best_cluster_idx = None
#         for i, cluster_vec in enumerate(cluster_embeddings):
#             sim = cosine_similarity(doc_vector, cluster_vec)
#             if sim > best_similarity:
#                 best_similarity = sim
#                 best_cluster_idx = i
#         if best_cluster_idx is not None:
#             clusters[best_cluster_idx]['docs'].append(doc)
#             matched_doc_indices.add(idx)
#             all_vectors = [np.array(embeddings.embed_query(d)) for d in clusters[best_cluster_idx]['docs']]
#             cluster_embeddings[best_cluster_idx] = np.mean(all_vectors, axis=0)

#     new_answer = ""
#     for cluster in clusters:
#         new_answer += f"{cluster['header']}\n\n"
#         for doc in cluster['docs']:
#             new_answer += f"\t* {doc}\n\n"
#         new_answer += "\n"
#     return new_answer

# ------------------ Streamlit UI ------------------

def main():
    st.title("Legal Clause Clustering with RAG")
    st.write(
        "This web app uses a Retrieval-Augmented Generation (RAG) approach to analyze a legal clause, "
        "retrieve similar clauses from a document store, and then lets you group them based on topics of your choosing."
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
            # Add the uploaded clauses to the vector store.
            documents = [Document(page_content=text) for text in texts]
            _ = vector_store.add_documents(documents=documents)
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
    
    # Initialize session state for retrieved clauses if not present.
    if "retrieved_docs" not in st.session_state:
        st.session_state["retrieved_docs"] = []

    if st.button("Run Analysis"):
        if not api_key:
            st.error("Please enter your API key in the sidebar.")
            return
        if not is_valid_text(user_clause):
            st.error("Please enter a valid legal clause.")
        else:
            with st.spinner("Retrieving similar clauses..."):
                retrieved_docs = retrieve(user_clause, k=k)
            docs = [doc.page_content for doc in retrieved_docs]
            st.session_state["retrieved_docs"] = docs  # Store in session state
            st.subheader("Retrieved Clauses")
            st.write(f"Number of clauses retrieved: {len(docs)}")
            for i, clause in enumerate(docs, start=1):
                with st.expander(f"Clause {i}"):
                    st.write(clause)
    
    # After showing the retrieved clauses, allow the user to group them by topics.
    if st.session_state["retrieved_docs"]:
        docs = st.session_state["retrieved_docs"]
        st.subheader("Group Clauses by Topics")
        topics_input = st.text_input("Enter topics to group the clauses (comma separated)", 
                                     value="Confidentiality, Non-disclosure, Trade secrets")
        if st.button("Group Clauses by Topics"):
            if not is_valid_text(topics_input):
                st.error("Please enter valid topics.")
            else:
                with st.spinner("Grouping clauses based on your topics..."):
                    print("clustering")
                    topic_answer = generate_topics(topics_input, docs)
                    print("post processing")
                    topic_answer = post_process(topic_answer, docs)
                    print("done")
                st.subheader("Generated Clustering Answer (Based on Your Topics)")
                st.write(topic_answer)

if __name__ == "__main__":
    main()
