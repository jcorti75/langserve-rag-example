# ##############################################################################################################################################
# ################################################ Version Created 2025/03/06 ##################################################################
# ##############################################################################################################################################

from typing import Dict
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langserve import add_routes
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from pydantic import BaseModel, Field

# Step 1: Cargar variables de entorno
print("\n🛠️ Step 1: Cargando variables de entorno...")
load_dotenv(override=True)

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("🚨 ERROR: No se encontró OPENAI_API_KEY en el archivo .env")

print(f"✅ OPENAI_API_KEY: {OPENAI_API_KEY[:8]}")
print(f"✅ QDRANT_URL: {QDRANT_URL}")
print(f"✅ QDRANT_API_KEY: {QDRANT_API_KEY[:8] if QDRANT_API_KEY else '🚨 No encontrado'}")

# Step 2: Cargar archivos de texto y PDF
print("\n🛠️ Step 2: Cargando archivos...")
TEXT_PATH = os.path.abspath("state_of_the_union.txt")
PDF_PATH = os.path.abspath("Manuscripts_Speaking_The_History_of_Read.pdf")

text_loader = TextLoader(TEXT_PATH, encoding="utf-8")
pdf_loader = PyPDFLoader(PDF_PATH)

documents_text = text_loader.load()
documents_pdf = pdf_loader.load()

documents = documents_text + documents_pdf
print(f"✅ Archivos cargados correctamente. TXT: {len(documents_text)}, PDF: {len(documents_pdf)}")

# Step 3: Procesar documentos
print("\n🛠️ Step 3: Dividiendo documentos en fragmentos...")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
data = text_splitter.split_documents(documents)
print(f"✅ Documentos divididos en {len(data)} fragmentos.")

# Step 4: Inicializar OpenAI Embeddings
print("\n🛠️ Step 4: Inicializando OpenAI Embeddings...")
embeddings = OpenAIEmbeddings()
print("✅ OpenAI Embeddings inicializados correctamente.")

# Step 5: Conectar a Qdrant
print("\n🛠️ Step 5: Conectando a Qdrant...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
collections = client.get_collections()
collection_names = [col.name for col in collections.collections]

if "rag_generator" not in collection_names:
    client.create_collection(
        collection_name="rag_generator",
        vectors_config={"size": 1536, "distance": "Cosine"}
    )
    print("✅ Colección creada con éxito.")
else:
    print("✅ La colección 'rag_generator' ya existe.")

# Step 6: Agregar documentos al vector store
print("\n🛠️ Step 6: Agregando documentos a Qdrant...")
qdrant = QdrantVectorStore.from_documents(
    data,
    embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=True,
    collection_name="rag_generator",
    force_recreate=False
)
retriever = qdrant.as_retriever(search_kwargs={"k": 4})
print("✅ Documentos agregados a Qdrant.")

# Step 7: Verificar almacenamiento en Qdrant
print("\n🛠️ Step 7: Verificando documentos en Qdrant...")
response = client.count(collection_name="rag_generator")
print(f"✅ Número de documentos en Qdrant: {response.count}")

# Step 8: Configurar el modelo de lenguaje (LLM)
print("\n🛠️ Step 8: Configurando LLM de OpenAI...")
llm = ChatOpenAI(model="gpt-4o")
print("✅ LLM de OpenAI configurado correctamente.")

# Step 9: Configurar el Prompt con Evaluación de Calidad
print("\n🛠️ Step 9: Configurando prompt con evaluación de calidad...")
class ResponseEvaluation(BaseModel):
    score: float = Field(..., description="Puntaje entre 0 y 1")
    feedback: str = Field(..., description="Retroalimentación sobre la calidad de la respuesta")
    confidence: float = Field(..., description="Nivel de confianza en la respuesta entre 0 y 1")



prompt = ChatPromptTemplate.from_template("""

Tienes 2 funciones: 

A ) Eres un experto en evaluación de respuestas. Evalúa la calidad de la respuesta considerando:

1. Relevancia (30%)
2. Precisión (30%)
3. Claridad (20%)
4. Completitud (20%)

🔹 Pregunta: {question}
🔹 Contexto: {context}
🔹 Respuesta: {response}

Devuelve:
- score: Un número entre 0 y 1
- feedback: Comentario detallado sobre la calidad de la respuesta
- confidence: Nivel de confianza (0-1)

B) Respondes preguntas en base a :

Utilice el siguiente contexto recuperado para responder la pregunta.
Si no sabes la respuesta, dilo.
Responda en tres oraciones concisas.

Rechazar y explicar si la pregunta es:
- Ofensivo: Sólo 3 frases en este caso, sea específico
- No relacionado con el tema: solo 3 oraciones en este caso, sea específico 
- Tipo inapropiado: solo 3 oraciones en este caso, sea específico

Answer:
""")

print("✅ Step 10a: Evaluación configurada correctamente.")

# Step 10: Configurar la cadena con RunnablePassthrough
def format_docs(data):
    return "\n\n".join(doc.page_content for doc in data)

base_chain = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])), response=(lambda x: x["response"]))
    | prompt
    | llm
    | StrOutputParser()
)


class RemoveIntermediateSteps(RunnableParallel):
    def invoke(self, inputs: Dict):
        result = super().invoke(inputs)
        return result["output"]

rag_chain = RemoveIntermediateSteps(
    {"context": retriever, "question": RunnablePassthrough(), "response":RunnablePassthrough()}
) | base_chain

print("✅ Step 10b: Cadena base configurada correctamente.")

# Step 11: Configurar la API con FastAPI
print("\n🛠️ Step 11: Configurando la API con FastAPI...")
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Summarization App",
)
add_routes(app, rag_chain, path="/openai")
print("✅ API configurada correctamente.")

# Step 12: Iniciar el servidor
print("\n🛠️ Step 12: Iniciando el servidor en localhost:8080...")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)
    print("✅ Servidor iniciado correctamente.")
