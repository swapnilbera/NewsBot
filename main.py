import os
import streamlit as st
import time
from langchain.llms import Cohere
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()  # Take environment variables from .env

st.title("NewsBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_cohere.index"  # Change file extension to .index

main_placeholder = st.empty()
llm = Cohere(temperature=0.9, max_tokens=500)

# Define embeddings outside the if block
embeddings = CohereEmbeddings(
    cohere_api_key="6CM338B6WfQkasvpz27LDq8ZPGI1flFdqI7U4EEw",
    user_agent="langchain-app"  # Add this line
)

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data and add metadata (source URL) to each chunk
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Add metadata (source URL) to each document
    for doc in docs:
        doc.metadata["source"] = doc.metadata.get("source", "Unknown")

    # Create embeddings and save it to FAISS index
    vectorstore_cohere = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to disk
    vectorstore_cohere.save_local(file_path)
    main_placeholder.text("FAISS index saved to disk...✅✅✅")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        # Load the FAISS index from disk
        vectorstore = FAISS.load_local(
            file_path,
            embeddings,
            allow_dangerous_deserialization=True  # Add this line
        )
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        
        # Result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            # Extract source URLs from the metadata
            source_urls = set()
            for doc in vectorstore.similarity_search(query):
                source_urls.add(doc.metadata.get("source", "Unknown"))
            
            # Display the source URLs
            for url in source_urls:
                st.write(url)