import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub, GPT4All
from langchain.callbacks import StdOutCallbackHandler

def get_text_from_docs(docs):
    text = ""
    for doc in docs:
        file_extension = doc.name.split('.')[-1]
        if file_extension == "pdf":
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file_extension == "csv":
            st.write(f"You uploaded a CSV file: {doc.name}")
        elif file_extension == "txt":
            text += doc.read().decode()
        else:
            st.write(f"Unsupported file format: {doc.name}. Please upload a PDF, CSV, or TXT file.")
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":2048})
    # llm = GPT4All(model="/Users/diegosilva/Library/Application Support/nomic.ai/GPT4All/ggml-gpt4all-j-v1.3-groovy.bin",
    #             max_tokens=2000,
    #             backend="gptj",
    #             verbose=True,
    #             callbacks=[StdOutCallbackHandler()],
    #             n_threads=8,  # Change this according to your cpu threads
    #             temp=0.5,)  

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple Documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple documents :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents PDF/CSV/TXT")
        docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", type=["pdf", "csv", "txt"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get get texts
                raw_text = get_text_from_docs(docs)
                st.write(raw_text)
            with st.spinner("Getting text chunks..."):
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
            with st.spinner("Getting vector store..."):
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
            with st.spinner("Getting conversation chain..."):
                # create conversation chain  
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
