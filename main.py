import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

# ============================
# Hugging Face API token
# ============================
load_dotenv()
hf_token = os.getenv('HuggingFaceToken')
if hf_token:
    os.environ['HUGGINGFACE_TOKEN'] = hf_token
else:
    print("Warning: HuggingFaceToken environment variable not set")



# ── Global variables ─────────────────────────
# These store your RAG components after startup
retriever = None
generator = None

# ── Startup logic (Concept 6) ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, generator


    # ============================
    # Step 2: Load your documents
    # ============================
    print("Loading documents...")
    documents = TextLoader('data/sample.txt').load()
    print(f'Loaded {len(documents)} documents.')
    

    # ============================
    # Step 3: Split documents into chunks
    # ============================
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # ============================
    # Step 4: Create embeddings - We convert text chunks into vectors
    # ============================
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # ============================
    # Step 5: Create vector store (FAISS)
    # ============================
    print("Building vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})



    # ============================
    # Step 6: Load LLM via HF Inference API
    # ============================
    print("Connecting to LLM...")
    generator = InferenceClient(
        model="openai/gpt-oss-120b",
        token=hf_token
    )

    print("RAG ready!")
    yield
    # shutdown
    print("Shutting down...")

# ── Create app ────────────────────────────────
app = FastAPI(lifespan=lifespan)

# ── Request and Response models ───────────────
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

# ── Endpoints ─────────────────────────────────────────────

# simple GET to check server alive
@app.get("/health")
def health():
    return {"status": "ok"}

# POST to send question
# error handling
@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):

    # error handling — validate input
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(request.question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Call LLM
    response = generator.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "Answer using ONLY the context. If not in context say I don't know."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{request.question}"
            }
        ],
        max_tokens=200,
        temperature=0.7,
    )

    answer = response.choices[0].message.content

    return AnswerResponse(
        question=request.question,
        answer=answer
    )

# ── Run directly ──────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # Run with: python main.py or uvicorn main:app --reload