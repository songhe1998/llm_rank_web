__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
# New: Function to prepare indexed clauses for the LLM
def prepare_indexed_clauses(docs):
    indexed_clauses = []
    for i, doc in enumerate(docs, start=1):
        # Create a shortened preview of the clause (first 60 chars)
        preview = doc[:60] + "..." if len(doc) > 60 else doc
        indexed_clauses.append(f"[{i}] {preview}")
    return indexed_clauses

# New: Modified topic prompt that works with indices
topic_prompt_template = """
Analyze the following retrieved clauses and categorize them into distinct clusters based on the provided topics. The topics to group by are: {topics}.

Each clause has an index number in square brackets []. For each topic, list ONLY THE INDICES of the clauses that relate to it. If a clause does not clearly belong to any of the provided topics, include its index in a separate cluster labeled [Other].

Format your response as follows:
[Topic Name]: [Brief explanation of how the clauses relate to the topic]
    Indices: [1, 5, 8, 12, ...]
    
[Other]: [Explanation for clauses not fitting any topic]
    Indices: [2, 7, 10, ...]

Ensure that every retrieved clause index is included in one of the clusters.
Retrieved clauses:
{context}
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

# Updated: Generate topic-based clustering using provided topics and indices
def generate_topics(topics: str, docs: list):
    indexed_clauses = prepare_indexed_clauses(docs)
    docs_content = "\n".join(indexed_clauses)
    prompt_message = topic_prompt_template.format(topics=topics, context=docs_content)
    messages = [{"role": "user", "content": prompt_message}]
    response = llm.invoke(messages)
    return response.content

# Validate if text is non-empty
def is_valid_text(text):
    return isinstance(text, str) and text.strip() != ""


def parse_text(text):
    clusters = {}
    # Split text into blocks assuming each cluster is separated by a blank line
    blocks = text.strip().split("\n\n")
    
    for block in blocks:
        # Get non-empty lines in the block
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        
        # First line should contain the topic name and description separated by ':'
        topic_line = lines[0]
        parts = topic_line.split(":", 1)
        if len(parts) < 2:
            continue  # Skip block if format is unexpected
        
        # Clean up the topic name by removing bold markers and extra spaces
        topic_name_raw = parts[0].strip()
        topic_name = topic_name_raw.strip("*")
        
        # The description is the remainder of the first line after the colon
        topic_description = parts[1].strip()
        
        # Find the line that contains the indices (could be any of the following lines)
        indices_line = None
        for line in lines:
            if "Indices:" in line:
                indices_line = line
                break
        if indices_line is None:
            continue  # Skip if no indices found
        
        # Split the indices line by colon and extract the list part
        indices_parts = indices_line.split(":", 1)
        if len(indices_parts) < 2:
            continue
        indices_str = indices_parts[1].strip()
        # Remove the surrounding brackets and split by comma to convert to integers
        indices_str = indices_str.strip("[]")
        indices = [int(x.strip()) for x in indices_str.split(",") if x.strip()]
        
        # Save in dictionary: topic name as key, and a dict with description and indices as value
        clusters[topic_name] = {
            "description": topic_description,
            "indices": indices
        }
    
    return clusters

# New: Process a subset of documents (from a cluster) for sub-clustering
def sub_cluster(subtopics: str, docs: list, indices: list):
    # Extract only the documents that belong to the selected cluster
    subset_docs = [docs[idx-1] for idx in indices if 1 <= idx <= len(docs)]
    
    # Prepare indexed clauses for the subset
    indexed_subset = prepare_indexed_clauses(subset_docs)
    docs_content = "\n".join(indexed_subset)
    
    # Use the same prompt but with the subtopics
    prompt_message = topic_prompt_template.format(topics=subtopics, context=docs_content)
    messages = [{"role": "user", "content": prompt_message}]
    response = llm.invoke(messages)
    
    # Process the response and map back to original indices
    subclusters = parse_text(response.content)
    
    # Map the sub-indices back to the original document indices
    mapped_subclusters = {}
    for subtopic, data in subclusters.items():
        # Map each sub-index back to original index
        original_indices = [indices[idx-1] for idx in data['indices'] if 1 <= idx <= len(indices)]
        mapped_subclusters[subtopic] = {
            "description": data['description'],
            "indices": original_indices
        }
    
    return mapped_subclusters, response.content

# Updated: Process LLM output with indices and replace with full text
def post_process(answer, docs):
    print(answer)
    
    clusters = parse_text(answer)
    new_answer = ""
    
    # Track all unique indices across all clusters
    all_indices = set()
    
    # Iterate through each topic in the clusters
    for topic_name, topic_data in clusters.items():
        # Add topic name and description
        new_answer += f"[{topic_name}]: {topic_data['description']}\n\n"
        
        # Track number of clauses in this topic
        topic_indices = topic_data['indices']
        
        # Add all indices to our tracking set
        all_indices.update(topic_indices)
        
        # Add each clause referenced by the indices
        for idx in topic_indices:
            if 1 <= idx <= len(docs):  # Check if index is valid
                clause_text = docs[idx-1]  # Get the full text (indices are 1-based)
                new_answer += f"\t* {clause_text}\n\n"
            else:
                new_answer += f"\t* (Invalid index: {idx})\n\n"
        
        # Add extra spacing between topics
        new_answer += "\n"
    
    # Print statistics about the clustering
    print(f"Total unique indices: {len(all_indices)}")
    print(f"Total documents: {len(docs)}")
    print(f"Percentage covered: {(len(all_indices) / len(docs)) * 100:.2f}%")
    
    # Add statistics to the answer
    new_answer += f"Statistics:\n"
    new_answer += f"- Total unique clauses categorized: {len(all_indices)}\n"
    new_answer += f"- Total clauses in dataset: {len(docs)}\n"
    new_answer += f"- Coverage: {(len(all_indices) / len(docs)) * 100:.2f}%\n"
    
    # If no clusters were parsed, return the original answer
    if not clusters:
        return "Could not parse clustering result properly:\n\n" + answer
    
    return new_answer, clusters


# ------------------ Streamlit UI ------------------

def main():
    st.title("Legal Clause Clustering with RAG")
    st.write(
        "This web app uses a Retrieval-Augmented Generation (RAG) approach to analyze a legal clause, "
        "retrieve similar clauses from a document store, and then lets you group them based on topics of your choosing."
    )
    
    # Initialize session states if they don't exist
    if "retrieved_docs" not in st.session_state:
        st.session_state["retrieved_docs"] = []
    if "clusters" not in st.session_state:
        st.session_state["clusters"] = {}
    if "subclusters_result" not in st.session_state:
        st.session_state["subclusters_result"] = None
    if "cluster_result" not in st.session_state:
        st.session_state["cluster_result"] = None
    
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
        'For the purposes of this Agreement, "Confidential Information" means any non-public, proprietary, '
        'or sensitive information disclosed by one party ("Disclosing Party") to the other party ("Receiving Party"), '
        'whether in written, oral, electronic, or other tangible or intangible form, including but not limited to '
        'business plans, financial data, customer lists, trade secrets, and proprietary technology.'
    )
    user_clause = st.text_area("Enter a legal clause to analyze:", value=default_clause, height=150)
    
    # Slider for choosing the number of documents to retrieve
    k = st.sidebar.slider("Number of documents to retrieve", min_value=10, max_value=100, value=50, step=5)
    
    # Main Analysis Button
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
            st.session_state["cluster_result"] = None  # Reset cluster result
            st.session_state["subclusters_result"] = None  # Reset subcluster result
            
            # Show retrieved clauses
            st.subheader("Retrieved Clauses")
            st.write(f"Number of clauses retrieved: {len(docs)}")
            for i, clause in enumerate(docs, start=1):
                with st.expander(f"Clause {i}"):
                    st.write(clause)
    
    # Only show clustering UI if we have documents
    if st.session_state["retrieved_docs"]:
        docs = st.session_state["retrieved_docs"]
        
        # First level clustering
        st.subheader("Group Clauses by Topics")
        topics_input = st.text_input("Enter topics to group the clauses (comma separated)", 
                                   value="Confidentiality, Non-disclosure, Trade secrets",
                                   key="main_topics")
        
        # First level clustering button
        if st.button("Group Clauses by Topics", key="main_cluster_button"):
            if not is_valid_text(topics_input):
                st.error("Please enter valid topics.")
            else:
                with st.spinner("Grouping clauses based on your topics..."):
                    topic_answer = generate_topics(topics_input, docs)
                    formatted_answer, clusters = post_process(topic_answer, docs)
                    st.session_state["clusters"] = clusters
                    st.session_state["cluster_result"] = formatted_answer
                    # Reset subcluster result when main clustering changes
                    st.session_state["subclusters_result"] = None
        
        # Display clustering result if available
        if st.session_state.get("cluster_result"):
            st.subheader("Generated Clustering Answer (Based on Your Topics)")
            st.write(st.session_state["cluster_result"])
            
            # Sub-clustering section
            st.subheader("Sub-Clustering")
            st.write("Select a cluster for further topic-based sub-clustering:")
            
            # Get available clusters
            cluster_options = list(st.session_state["clusters"].keys())
            
            # Only show subcluster options if we have clusters
            if cluster_options:
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    selected_cluster = st.selectbox(
                        "Choose a cluster",
                        cluster_options,
                        key="selected_cluster"
                    )
                
                with col2:
                    subcluster_topics = st.text_input(
                        "Enter sub-topics (comma separated)",
                        key="subtopics"
                    )
                
                # Sub-clustering button
                if st.button("Generate Sub-Clusters", key="subcluster_button"):
                    if not is_valid_text(subcluster_topics):
                        st.error("Please enter valid sub-topics.")
                    else:
                        # Get the indices for the selected cluster
                        cluster_indices = st.session_state["clusters"][selected_cluster]["indices"]
                        
                        with st.spinner(f"Creating sub-clusters for '{selected_cluster}'..."):
                            subclusters, raw_response = sub_cluster(
                                subcluster_topics,
                                docs,
                                cluster_indices
                            )
                            
                            # Create a formatted display for subclusters
                            subclusters_display = f"## Sub-clusters for [{selected_cluster}]\n\n"
                            subclusters_display += f"*{st.session_state['clusters'][selected_cluster]['description']}*\n\n"
                            
                            for subtopic, data in subclusters.items():
                                subclusters_display += f"### [{subtopic}]: {data['description']}\n\n"
                                for idx in data['indices']:
                                    if 1 <= idx <= len(docs):
                                        subclusters_display += f"* {docs[idx-1]}\n\n"
                                    else:
                                        subclusters_display += f"* (Invalid index: {idx})\n\n"
                            
                            # Store the formatted subclusters display
                            st.session_state["subclusters_result"] = subclusters_display
                
                # Display subcluster results if available
                if st.session_state.get("subclusters_result"):
                    st.markdown(st.session_state["subclusters_result"])

if __name__ == "__main__":
    main()
