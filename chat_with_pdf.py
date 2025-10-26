import streamlit as st
import os
from openai import OpenAI
from os import environ

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# PDF support
from langchain_community.document_loaders import PyPDFLoader  
import tempfile  

# Get API key from environment variable
environ["OPENAI_API_KEY"] = os.environ.get("API_KEY")
environ["OPENAI_BASE_URL"] = "https://api.ai.it.cornell.edu"

# Set models to use for chat and embedding and temperature
CHAT_MODEL = "openai.gpt-4o"
EMBED_MODEL = "openai.text-embedding-3-large"
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)

# UI Title
st.title("📝 File Q&A with OpenAI")

# Label file types supported, accept multiple files
uploaded_file = st.file_uploader("Upload an article BEFORE chat", type=("txt", "md", "pdf"), accept_multiple_files=True) 
question = st.chat_input(
    "Ask something about the article",
    disabled=not uploaded_file,
)
# Keeps message history and displays messages
if "messages" not in st.session_state:
    # Rewrite the message to remind users to upload files first
    st.session_state["messages"] = [{"role": "assistant", "content": "Please upload one or more files before asking a question about the article."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Combines retrieved document chunks into a single string
def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

# RAG process
if question and uploaded_file:
    # Read uploaded text
    documents = []  # Document list created to support multiple files
    for file in uploaded_file:  
        # pdf files
        if file.name.lower().endswith(".pdf"): 
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:  
                tmp.write(file.read())  
                tmp_path = tmp.name  
            loader = PyPDFLoader(tmp_path)  
            documents.extend(loader.load())  
        # txt files
        else:  
            text = file.read().decode("utf-8")  
            documents.append(Document(page_content=text, metadata={"source": file.name}))  

    # Split text into chunks and generate embeddings for each chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    chunks = splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings(model=EMBED_MODEL))

    # Prompt template provided
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Question: {question} 
    
    Context: {context} 

    Answer:
    """
    
    prompt = PromptTemplate.from_template(template)

    # Retrieve the top 5 similar chunks to the question
    k = 5
    docs = vectorstore.similarity_search(question, k=k)

    # Build message sequence from retrived context and question
    context = format_docs(docs)
    messages = [
        SystemMessage(content=prompt.format(question=question, context=context)),
        HumanMessage(content=question),
    ]

    # Call the model
    response = llm.invoke(messages)
    answer = response.content

    # Display answer and retrieved chunks
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        st.write(answer)
        with st.expander("Sources (retrieved chunks)"):
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "(no source)")
                st.markdown(f"**[{i}]** `{src}`")
                st.code(d.page_content[:1000], language="markdown")

    st.session_state["messages"].append({"role": "assistant", "content": answer})