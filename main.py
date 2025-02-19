import os
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import time

#Loading env files
from dotenv import load_dotenv


#Loading all the environment variable
load_dotenv()

#Chat Model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=1,
    max_retries=3
)


st.title("News Research Tool 📈")

st.sidebar.title("News Article URLs")

urls = []
#Creating a URL using input
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_urls = st.sidebar.button("Processed")
file_name = "vectored_index.pkl"

#placeholder UI component
main_place_holder = st.empty()
is_error = False
#When the button is pressed
if process_urls:
    #Loading urls
    main_place_holder.text("Data Loading...Started...✅✅✅ ")
    try:
        url_loader = UnstructuredURLLoader(urls=urls)

        data = url_loader.load()

        #Splitting data into chunks
        main_place_holder.text("Text Splitter...Started...✅✅✅ ")

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n","\n","."," "],
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(data)

        #Embedding data using huggingface
        main_place_holder.text("Data Embedding...Started...✅✅✅ ")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        #Creating a fiass vector index
        vector_faiss_index = FAISS.from_documents(chunks,embeddings)
        time.sleep(2)

        #Index is ready and now saving the database
        with open(file_name,"wb") as file:
            pickle.dump(vector_faiss_index,file)
            print("THe file has been saved!")

    except:
        is_error=True
        st.write("**Some Error has occurred!**")
        st.write("Please suppy URLS with text data or you can reboot the application.")
#Now Creating a Question box which allows user to ask questions
query =""
if not is_error:
    query = main_place_holder.text_input("Question : ")

if query:
    #Loading database
    if os.path.exists(file_name):
        with open(file_name,"rb") as file:
            vectorIndex = pickle.load(file)


    #Creating a retrieval QA chain
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorIndex.as_retriever())

    #Asking Chain about query
    response = chain({"question":query},return_only_outputs=True)
    print(response)
    st.header("Answer")
    st.write(response["answer"])
    sources = response.get("sources","")
    if sources:
        st.subheader("Sources")
        source_list = sources.split("\n")
        for source in source_list:
            st.write(source)

