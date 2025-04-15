import os
import io
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from PyPDF2 import PdfReader
from pinecone import Pinecone
from dotenv import load_dotenv
import streamlit as st
from thefuzz import fuzz

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

# --- CONFIGURACIÓN INICIAL ---
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
        messages_for_api = []
        
        # Procesar todos los mensajes manteniendo el system message original
        for message in input.messages:
            if message.type == "system":
                messages_for_api.append({"role": "system", "content": message.content})
            elif message.type == "human":
                # Corregir variaciones del nombre
                corrected_content = message.content.replace("Donadelo", "Donadello") if fuzz.ratio("Donadelo", "Donadello") > 80 else message.content
                messages_for_api.append({"role": "user", "content": corrected_content})
            elif message.type == "ai":
                messages_for_api.append({"role": "assistant", "content": message.content})

        # Llamar a la API
        response = client.chat.completions.create(
            messages=messages_for_api,
            model=self.model,
            temperature=self.temperature,
            max_tokens=4096,
            top_p=1,
        )

        return response.choices[0].message.content


def init_langchain_clients():
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index = 'gptsmallembedding'

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MPNet-base-v2"
    )

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index)
    
    return PineconeVectorStore(index=index, embedding=embeddings)


def download_document(file_name):
    with open(file_name, 'rb') as f:
        pdf_content = f.read()

    pdf_reader = PdfReader(io.BytesIO(pdf_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Preprocesamiento de texto
    text = text.replace("\n", " ").replace("\\n", " ")
    text = " ".join(text.split())  # Eliminar espacios múltiples
    return text


def split_text_with_langchain(text, max_length, chunk_overlap, source):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    documents = text_splitter.create_documents([text])
    for doc in documents:
        doc.metadata = {"source": source}
    indices = [f"CV_{i+1}" for i in range(len(documents))]
    return documents, indices


def cargar_embeddings_si_no_existen(pinecone_vectorstore):
    try:
        index_stats = pinecone_vectorstore._index.describe_index_stats()
        if index_stats["total_vector_count"] == 0:
            file_name = 'Curriculum.pdf'
            text = download_document(file_name)
            documents, indices = split_text_with_langchain(
                text, 
                max_length=500, 
                chunk_overlap=50, 
                source='CV'
            )
            pinecone_vectorstore.add_documents(documents=documents, ids=indices)
            st.success("Embeddings cargados exitosamente!")
    except Exception as e:
        st.error(f"Error cargando embeddings: {e}")


def main_request():
    st.title("Asistente de CV Profesional")
    st.write("Experto en análisis de currículums")

    pinecone_vectorstore = init_langchain_clients()
    cargar_embeddings_si_no_existen(pinecone_vectorstore)

    # Configurar retriever con MMR
    retriever = pinecone_vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "lambda_mult": 0.25}
    )
    
    llm = CustomAzureChatModel(model="gpt-4o-mini", temperature=0.5)

    memory = ConversationBufferWindowMemory(
        k=3,
        return_messages=True,
        memory_key="historial_chat"
    )

    system_prompt = """Eres un experto en análisis de CVs. Responde ÚNICAMENTE con información del contexto proporcionado.

CONTEXTO:
{context}

Si la pregunta no puede responderse con el contexto, di explícitamente: 'No hay información relevante en el CV'"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="historial_chat"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)

    if 'memory' not in st.session_state:
        st.session_state.memory = memory

    pregunta = st.text_input("Haz tu pregunta:")

    if pregunta:
        # Corregir posible error en el nombre
        pregunta_procesada = pregunta.replace("Donadelo", "Donadello") if fuzz.ratio("Donadelo", "Donadello") > 80 else pregunta
        
        memory_variables = st.session_state.memory.load_memory_variables({})
        inputs = {
            "input": pregunta_procesada,
            "historial_chat": memory_variables["historial_chat"]
        }

        respuesta = rag_chain.invoke(inputs)



        st.markdown(f"**Respuesta:** {respuesta['answer']}")
        # Mostrar contexto para depuración
        st.write("Documentos relevantes:", [doc.page_content[:100].replace("\n", " ") + "..." for doc in respuesta["context"]])

        st.session_state.memory.save_context(
            {"input": pregunta}, 
            {"output": respuesta["answer"]}
        )

        st.divider()
        st.caption("Detalles técnicos:")
        st.write(respuesta)

if __name__ == "__main__":
    main_request()