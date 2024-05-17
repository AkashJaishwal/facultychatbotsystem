import logging
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import sys

app = Flask(__name__)
CORS(app)

# Define the path to the FAISS knowledge base
DB_FAISS_PATH = "/vectorstore/db_faiss"

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@app.route("/")
def root():
    return "Welcome to the Faculty Bot Web Application!"


@app.route("/chat_csv", methods=["POST"])
def ask():
    query = request.json.get("query")
    logger.info("Received query: %s", query)

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Load the CSV data
        loader = CSVLoader(
            file_path="./data/faculty.csv",
            encoding="utf-8",
            csv_args={"delimiter": ","},
        )
        data = loader.load()
        print(data)

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(data)

        # Convert the text chunks into embeddings and save them to the FAISS knowledge base
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        db = FAISS.from_documents(text_chunks, embeddings)
        db.save_local(DB_FAISS_PATH)

        # Initialize the language model and the conversational retrieval chain
        llm = CTransformers(
            model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
            model_type="llama",
            max_new_tokens=512,
            temperature=0.1,
        )
        qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        # Get the response from the conversational retrieval chain
        chat_history = []
        result = qa({"question": query, "chat_history": chat_history})
        response = {"answer": result["answer"]}

        logger.info("Response: %s", response)
        return jsonify(response), 200
    except Exception as e:
        logger.exception("Error occurred while processing query")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)


# from langchain.document_loaders.csv_loader import CSVLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain_community.llms import CTransformers
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# import sys

# DB_FAISS_PATH = "/vectorstore/db_faiss"
# loader = CSVLoader(file_path="./data/faculty.csv", encoding="utf-8", csv_args={'delimiter': ','})
# data = loader.load()
# # print(data)

# # Split the text into Chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
# text_chunks = text_splitter.split_documents(data)

# # print(len(text_chunks))

# # Download Sentence Transformers Embedding From Hugging Face
# embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# # COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
# docsearch = FAISS.from_documents(text_chunks, embeddings)

# docsearch.save_local(DB_FAISS_PATH)


# # query = "What is the email of bindu agarwal"

# # docs = docsearch.similarity_search(query, k=3)

# # print("Result", docs)

# llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
#                     model_type="llama",
#                     max_new_tokens=300,
#                     temperature=0.001)

# qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

# while True:
#     chat_history = []
#     #query = "What is the value of  GDP per capita of Finland provided in the data?"
#     query = input(f"Input Prompt: ")
#     if query == 'exit':
#         print('Exiting')
#         sys.exit()
#     if query == '':
#         continue
#     result = qa({"question":query, "chat_history":chat_history})
#     print("Response: ", result['answer'])