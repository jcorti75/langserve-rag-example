##############################################################################################################################################
################################################ ULTIMA VERSION QUE FUNCIONA ##################################################################
##############################################################################################################################################
from typing import Dict
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langserve import add_routes
import os
from dotenv import load_dotenv

# Limpiar y cargar la clave API
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No se encontró OPENAI_API_KEY en el archivo .env")

print(f"Usando la clave API que comienza con: {OPENAI_API_KEY[:8]}...")

# Rutas de los documentos
TEXT_PATH = os.path.abspath("state_of_the_union.txt")
PDF_PATH = os.path.abspath("Manuscripts_Speaking_The_History_of_Read.pdf")

# Cargar y procesar los documentos
text_loader = TextLoader(TEXT_PATH, encoding="utf-8")
pdf_loader = PyPDFLoader(PDF_PATH)

# Combinar documentos
documents_text = text_loader.load()
documents_pdf = pdf_loader.load()
documents = documents_text + documents_pdf

# Dividir documentos en fragmentos manejables
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
data = text_splitter.split_documents(documents)

# Crear vectorstore y retriever
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(data, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4")

ANSWER_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you don't know the answer, say so.
Respond in three concise sentences.

Reject and explain if the question is:
- Offensive: Only 3 sentences in this case, be specific
- Not Related with the topic: Only 3 sentences in this case, be specific
- Inappropriate type: Only 3 sentences in this case, be specific

Always respond in Spanish.

Question: {question}
Context: {context}
Answer:
"""

prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

def format_docs(data):
    return "\n\n".join(doc.page_content for doc in data)

# Configurar la cadena base
base_chain = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])) )
    | prompt
    | llm
    | StrOutputParser()
)

# Modificar la cadena para eliminar pasos intermedios
class RemoveIntermediateSteps(RunnableParallel):
    def invoke(self, inputs: Dict):
        result = super().invoke(inputs)
        # Solo devolver la respuesta final
        return result["output"]

# Usar la clase personalizada para ocultar pasos intermedios
rag_chain = RemoveIntermediateSteps(
    {"context": retriever, "question": RunnablePassthrough()}
) | base_chain

# Crear app FastAPI
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Summarization App",
)

# Añadir rutas al servidor con el endpoint deseado
add_routes(app, rag_chain, path="/openai")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)


# from typing import Dict
# from fastapi import FastAPI
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_core.runnables import Runnable, RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain.text_splitter import CharacterTextSplitter
# from langserve import add_routes
# import os
# from dotenv import load_dotenv

# # Limpiar y cargar la clave API
# if "OPENAI_API_KEY" in os.environ:
#     del os.environ["OPENAI_API_KEY"]

# # load_dotenv(override=True)
# load_dotenv()

# # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# # if not OPENAI_API_KEY:
# #     raise ValueError("No se encontró OPENAI_API_KEY en el archivo .env")

# # print(f"Usando la clave API que comienza con: {OPENAI_API_KEY[:8]}...")

# # Cargar documentos y procesarlos
# DATA_PATH = os.path.abspath("state_of_the_union.txt")
# loader = TextLoader(DATA_PATH, encoding="utf-8")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# data = text_splitter.split_documents(documents)

# # Crear vectorstore y retriever
# embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.from_documents(data, embeddings)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# llm = ChatOpenAI(model="gpt-4")

# ANSWER_TEMPLATE = """
# You are an assistant for question-answering tasks.
# Use the following retrieved context to answer the question.
# If you don't know the answer, say so.
# Respond in three concise sentences.

# Reject and explain if the question is:
# - Offensive: Only 3 sentences in this case, be specific
# - Not Related with the topic: Only 3 sentences in this case, be specific
# - Inappropriate type: Only 3 sentences in this case, be specific

# Always respond in Spanish.

# Question: {question}
# Context: {context}
# Answer:
# """

# prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

# def format_docs(data):
#     return "\n\n".join(doc.page_content for doc in data)

# # Configurar la cadena base
# base_chain = (
#     RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # Clase para eliminar pasos intermedios completamente
# class SimplifiedChain(Runnable):
#     def __init__(self, chain):
#         self.chain = chain

#     def invoke(self, inputs: Dict):
#         try:
#             # Ejecutar la cadena base y devolver solo la salida final
#             result = self.chain.invoke(inputs)
#             if isinstance(result, dict):
#                 return result.get("output", "Error: No output generated")
#             return str(result)
#         except Exception as e:
#             # Capturar y devolver cualquier error inesperado
#             return f"Error procesando la solicitud: {str(e)}"

#     def get_input_schema(self):
#         return self.chain.get_input_schema()

#     def get_output_schema(self):
#         return self.chain.get_output_schema()

# rag_chain = SimplifiedChain(base_chain)

# # Crear app FastAPI
# app = FastAPI(
#     title="LangChain Server",
#     version="1.0",
#     description="Summarization App",
# )

# # Añadir rutas al servidor con el endpoint deseado
# add_routes(app, rag_chain, path="/openai")

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="localhost", port=8000)




