import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient


# ============================
# Step 1: Hugging Face API token
# ============================
load_dotenv()
hf_token = os.getenv('HuggingFaceToken')
if hf_token:
    os.environ['HUGGINGFACE_TOKEN'] = hf_token
else:
    print("Warning: HuggingFaceToken environment variable not set")


# ============================
# Step 2: Load your documents
# ============================
documents = TextLoader('data/sample.txt').load()
print(f'Loaded {len(documents)} documents.')



# ============================
# Step 3: Split documents into chunks
# ============================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# ============================
# Step 4: Create embeddings - We convert text chunks into vectors
# ============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ============================
# Step 5: Create vector store (FAISS)
# ============================
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 


# ============================
# Step 6: Load LLM via HF Inference API
# ============================
generator = InferenceClient(
    model="openai/gpt-oss-120b",
    token=hf_token
)

# ============================
# Step 7: Query loop
# ============================
while True:
    query = input("Ask something (or 'exit'): ")
    if query.lower() == "exit":
        break

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # ✅ Use chat_completion
    response = generator.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "Answer the question using ONLY the context provided. If the answer is not in the context, say 'I don't know'."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
        max_tokens=200,
        temperature=0.7,
    )

    answer = response.choices[0].message.content
    print("\nAnswer:\n", answer)
    print("-" * 50)