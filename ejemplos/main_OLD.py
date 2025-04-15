import os
import io
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from PyPDF2 import PdfReader
from pinecone import Pinecone
from dotenv import load_dotenv
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain_core.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder)
from langchain.prompts import SystemMessagePromptTemplate

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from openai import OpenAI


# --- CONFIGURACIÓN INICIAL
load_dotenv()
with open("API_KEYS.txt") as f:
    for line in f:
        key_value = line.strip().split("=")
        if len(key_value) == 2:
            key, value = key_value
            os.environ[key] = value

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)

from langchain_core.runnables import Runnable
from typing import Any, Optional
from langchain_core.runnables.config import RunnableConfig

class CustomAzureChatModel(Runnable):
    def __init__(self, model="gpt-4o-mini", temperature=0.7):
        self.model = model
        self.temperature = temperature

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        
        # Extraer el mensaje del usuario (el que tiene tipo "human")
        user_input = None
        for message in input.messages:
            if message.type == "human":
                user_input = message.content
                break

        # Extraer todos los mensajes anteriores con el formato adecuado
        context_msgs = []
        for message in input.messages:
            if message.type == "human":
                context_msgs.append({"role": "user", "content": message.content})
            elif message.type == "ai":
                context_msgs.append({"role": "assistant", "content": message.content})

        # Depurar el contenido de context_msgs
        print("Contexto construido:", context_msgs)

        # Crear la lista de mensajes, con el mensaje de "system" como primer elemento
        messages = [
            {"role": "system", "content": "Sos un asistente especializado en Curriculum Vitae."}
        ] + context_msgs + [{"role": "user", "content": user_input}]

        # Depurar el contenido de messages
        print("Mensajes enviados a la API:", messages)  

        # Llamar a la API de OpenAI para obtener la respuesta
        response = client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=4096,
            top_p=1,
        )

        # Retornar la respuesta del modelo
        ouput={"answer": response.choices[0].message.content}
        return ouput["answer"]


def init_langchain_clients():
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index = 'gptsmallembedding'

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MPNet-base-v2"     # Modelo con dimensión 1536
    )

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index)
    #print("INDEXXXXXXX")
    #print(index.describe_index_stats())

    pinecone_vectorstore = PineconeVectorStore(
        index=index, embedding=embeddings)

    return pinecone_vectorstore


def download_document(file_name):
    with open(file_name, 'rb') as f:
        pdf_content = f.read()

    pdf_reader = PdfReader(io.BytesIO(pdf_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    return text


def split_text_with_langchain(text, max_length, chunk_overlap, source):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length, chunk_overlap=chunk_overlap)
    metadata = [{"filename": source} for _ in range(len(text))]
    documents = text_splitter.create_documents([text], metadatas=metadata)
    indices = [f"{'CV'}_{i+1}" for i in range(len(documents))]
    return documents, indices


def cargar_embeddings_si_no_existen(pinecone_vectorstore):
    # TODO: Aquí podrías mejorar usando `describe_index_stats` para chequear si ya hay docs.
    try:
        # Limpiar Pinecone y subir el PDF
        file_name = 'Curriculum.pdf'
        text = download_document(file_name)
        documents, indices = split_text_with_langchain(text, 150, 0, 'CV')
        #pinecone_vectorstore._index.delete(delete_all=True)
        pinecone_vectorstore.add_documents(documents=documents, ids=indices)
    except Exception as e:
        st.error(f"Error cargando los embeddings: {e}")


def main_request():
    st.title("R.A.G. con memoria - NLP II")
    st.write("Asistente especializado")

    pinecone_vectorstore = init_langchain_clients()

    # Cargar automáticamente embeddings desde el PDF
    cargar_embeddings_si_no_existen(pinecone_vectorstore)

    retriever = pinecone_vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = CustomAzureChatModel(model="gpt-4o-mini", temperature=0.7)

    memory = ConversationBufferWindowMemory(
        k=5,
        return_messages=True,
        memory_key="historial_chat"
    )

    system_prompt = (
        "Sos un asistente especializado en Curriculum Vitae. "
        "Respondé en forma clara y concisa usando el contexto provisto. "
        "Si no hay información suficiente, respondé: 'No existe esa información'."
        "Contexto: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="historial_chat"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=combine_chain)

    if 'memory' not in st.session_state:
        st.session_state.memory = memory

    pregunta = st.text_input("Haz una pregunta:")

    if pregunta:
        memory_variables = st.session_state.memory.load_memory_variables({})
        inputs = {
            "input": pregunta,
            "historial_chat": memory_variables["historial_chat"]
        }

        respuesta = rag_chain.invoke(inputs)

        st.session_state.memory.save_context(
            {"input": pregunta}, {"output": respuesta["answer"]})

        st.text_area("Respuesta del Chatbot:",
                     value=respuesta["answer"], height=100)
        st.write("Respuesta completa del Chatbot:", respuesta)


if __name__ == "__main__":
    main_request()
