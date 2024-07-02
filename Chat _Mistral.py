import streamlit as st
import getpass
#from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_mistralai import ChatMistralAI
os.environ["MISTRAL_API_KEY"] = "jO8DC8DXDjgSLHsJPopNgtejjB5myZKC"
os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(separator='\n',chunk_size=1000,chunk_overlap=200)
    chunks=text_splitter.split_text(raw_text)
    return chunks

def get_vector_database(text_chunks):
    embeddings=MistralAIEmbeddings()
    vectordb=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectordb

def get_rag_chain(retriever):
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
    )
    llm = ChatMistralAI(model_name="mistral-large-latest")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def handle_userinput(user_question):
    #final_answer=""
    
    #for i in range(9):
    #while(len(final_answer)<1000):
    response=st.session_state.rag.invoke({"input":user_question})
        #final_answer+=response['answer']
        #user_question=final_answer
    
    
    st.write(response["answer"])




def main():
    #load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    st.header("Chat with your PDFs :))")
    user_question=st.text_input("Ask a question from these PDFs")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs=st.file_uploader("Upload your PDFs here",accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                #get_pdf_text
                raw_text=get_pdf_text(pdf_docs)
                

                #get_text_chunks
                text_chunks=get_text_chunks(raw_text)
                

                #get_vector database
                vectordb=get_vector_database(text_chunks)
                st.write("Processed Successfully !")
                retriever=vectordb.as_retriever()

                #get conversation chain
                st.session_state.rag=get_rag_chain(retriever)


if __name__=='__main__':
    main()