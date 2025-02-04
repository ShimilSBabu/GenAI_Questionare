# -*- coding: utf-8 -*-
import os

# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import StrOutputParser
from langchain.schema.messages import SystemMessage, HumanMessage
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from torch.cuda.tunable import read_file

# os.environ["LANGCHAIN_TRACING_V2"] = "true"


current_dir = os.getcwd()
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

os.makedirs(persistent_directory, exist_ok=True)

"""# Create embeddings"""
#
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

"""# Define the messages for the model"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human" , "Here are some documents that should be used for preparing a questionare on the topic: {topic}"
        + "\nPrepare a questionare containing {num_questions} questions and its answers. The questionare should be in a multiple choice pattern."
        + "\n\nRelevant Documents:\n {joined_relevant_docs}"
        + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
        + "\n\nI need the output in html format."
        + "\n Each question should have 4 options"
        # + " and the correct answer should only get displayed after an option is selected for that question."
        + "\nOnly the html code is needed as thee. Do not include parts of conversation in the output."
        + "\nDo not include '```html' and '</html> ```' parts."
         )

])

"""# Retrieve relevant documents based on the query

## Defining the Retriever
"""

def create_db_from_uploaded_file(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Convert pages into text format for vector storage
    documents = [page.page_content for page in pages]

    # Initialize ChromaDB with embeddings
    db = Chroma.from_texts(documents, embedding=embeddings, persist_directory=persistent_directory)

    # Check if file exists before deleting
    if os.path.exists(file_path):
        os.remove(file_path)
        print("File deleted successfully!")
    else:
        print("File not found for deletion.")


def retrive_relevent_docs(k=3, score_threshold=0.2, query=""):
    # Load the existing vector store with the embedding function
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embeddings)
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold}
    )

    """## Fetching only relevent documents using the retriever"""
    relevant_docs = retriever.invoke(query)
    return relevant_docs


def questionare_maker(api_key = '', topic = 'AI', num_questions = 3):
    os.environ["MISTRAL_API_KEY"] = str(api_key)

    model = ChatMistralAI(model="mistral-large-latest")
    chain = prompt_template | model | StrOutputParser()

    folder_path = os.path.join(os.getcwd(),'uploads')
    files = os.listdir(folder_path)
    try:
        file_path = os.path.join(folder_path, files[0])
    except:
        return "Kindly upload a PDF."
    create_db_from_uploaded_file(file_path)
    print("DB Created")

    relevant_docs = retrive_relevent_docs(k=6, score_threshold=0.3, query= topic)
    joined_relevant_docs = "\n\n".join([doc.page_content for doc in relevant_docs])

    result = chain.invoke({"topic": topic, "num_questions": num_questions, "joined_relevant_docs": joined_relevant_docs})

    return result