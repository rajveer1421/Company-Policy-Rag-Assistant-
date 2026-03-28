# 🏢 Company Policy RAG Assistant

An intelligent question-answering system for employees built using **Retrieval-Augmented Generation (RAG)**. The assistant retrieves relevant information from company policy documents and generates accurate, hallucination-resistant answers using a language model.

---

## 🚩 Problem Statement

Employees often struggle to locate accurate information inside long policy documents such as:
- Leave rules & attendance policies
- Reimbursement & expense processes
- Workplace conduct & code of ethics
- Recruitment & onboarding guidelines
- Mobile phone & IT usage policies

Traditional keyword search is slow and requires manual reading through hundreds of pages.

---

## 💡 Solution

This project builds an AI-powered assistant that:
- ✅ Understands natural language queries
- ✅ Retrieves the most relevant policy passages
- ✅ Generates clear, grounded answers in natural language
- ✅ **Refuses to answer if query is irrelevant to company policy documents**
- ✅ **Prevents hallucination via strict PromptTemplate constraints**
- ✅ Cites source documents for transparency

---

## 🧠 Architecture

### Version 1 — DPR + GPT-2 Pipeline
```
User Query
    ↓
DPR Context Encoder → Document Embeddings
    ↓
FAISS Vector Search → Top-K Relevant Passages
    ↓
GPT-2 → Final Answer
```

### Version 2 — LangChain + LLaMA Pipeline ✅ (Current)
```
User Query
    ↓
TextLoader (langchain_community) → Raw Policy Text
    ↓
CharacterTextSplitter → Chunked Documents
    ↓
HuggingFaceEmbeddings (sentence-transformers) → Dense Vectors
    ↓
ChromaDB (Persisted Vector Store) → Top-K Relevant Passages
    ↓
PromptTemplate (Anti-Hallucination Guard) → Constrained Context
    ↓
LangChain RetrievalQA Chain
    ↓
LLaMA (via HuggingFace) → Grounded Final Answer
```

---

## ⭐ Non-Functional Improvements in V2

These were the core engineering challenges solved in Version 2:

### 🛡️ 1. Hallucination Prevention
A strict `PromptTemplate` instructs the model to only answer from retrieved context:
```python
prompt_template = """
Use the following context to answer the question.
If the answer is not found in the context, 
respond with: "I don't know the answer based on the 
available company policy documents."
Do NOT make up any information.

Context: {context}
Question: {question}
Answer:
"""
```
> This prevents the LLM from generating plausible-sounding but incorrect policy information.

### 📄 2. Intelligent Document Chunking
Using `CharacterTextSplitter` to split large policy documents into meaningful overlapping chunks:
```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```
> Overlap ensures context is never lost at chunk boundaries.

### 🗄️ 3. Persistent Vector Store
ChromaDB persists embeddings to disk so they don't need to be recomputed on every run:
```python
docsearch = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="./chroma_embeddings"
)
```
> Drastically reduces startup time on repeated use.

### 🔍 4. Semantic Search over Keyword Search
`HuggingFaceEmbeddings` with `sentence-transformers` captures semantic meaning:
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
```
> Finds relevant passages even when exact keywords don't match.

### 📥 5. Modular Document Loading
`TextLoader` from `langchain_community` makes swapping policy documents effortless:
```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("companyPolicies.txt")
document = loader.load()
```
> Easily extendable to PDFs, Word docs, or URLs in future.

---

## 🔄 Full V2 Pipeline Code Flow
```python
# Step 1 — Load Documents
from langchain_community.document_loaders import TextLoader
loader = TextLoader("companyPolicies.txt")
document = loader.load()

# Step 2 — Split into Chunks
from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(document)

# Step 3 — Generate Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

# Step 4 — Store in ChromaDB
from langchain.vectorstores import Chroma
docsearch = Chroma.from_documents(docs, embeddings,
                persist_directory="./chroma_embeddings")

# Step 5 — Anti-Hallucination PromptTemplate
from langchain.prompts import PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Use the following context to answer the question.
    If the answer is not found in the context, respond with:
    "I don't know the answer based on the available 
    company policy documents."
    Do NOT make up any information.

    Context: {context}
    Question: {question}
    Answer:"""
)

# Step 6 — RetrievalQA Chain with LLaMA
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template}
)

# Step 7 — Query
result = qa_chain({"query": "What is the leave policy?"})
print(result["result"])
```

---

## 🛠 Tech Stack

| Component | V1 | V2 ✅ Current |
|---|---|---|
| Document Loader | Custom | `TextLoader` (langchain_community) |
| Text Splitting | Manual | `CharacterTextSplitter` |
| Embeddings | DPR | `HuggingFaceEmbeddings` |
| Vector Store | FAISS | ChromaDB (Persistent) |
| Hallucination Guard | ❌ None | ✅ `PromptTemplate` |
| LLM | GPT-2 | LLaMA (HuggingFace) |
| Framework | PyTorch | LangChain |

---

## 📂 Project Structure
```
Company-Policy-Rag-Assistant/
│
├── companyPolicies.txt                         # Source policy documents
├── CompanyPolicyAgentUsingLangchain.ipynb      # Main notebook (V2)
├── chroma_embeddings/                          # Persisted ChromaDB vectors
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install langchain langchain-community chromadb 
pip install sentence-transformers transformers torch
```

### Run the Notebook
```bash
git clone https://github.com/rajveer1421/Company-Policy-Rag-Assistant-.git
cd Company-Policy-Rag-Assistant-
jupyter notebook CompanyPolicyAgentUsingLangchain.ipynb
```

## 📊 Policies Covered

- 📋 Code of Conduct
- 🏖️ Leave & Attendance Policy
- 💰 Reimbursement Policy
- 📱 Mobile Phone Policy
- 🤝 Recruitment Policy
- 🔒 Data Privacy Policy

---

## 🔮 Future Improvements

- [ ] Streamlit / Gradio web interface
- [ ] PDF & Word document ingestion
- [ ] Multi-turn conversation memory
- [ ] REST API deployment
- [ ] Slack / Teams integration

---
