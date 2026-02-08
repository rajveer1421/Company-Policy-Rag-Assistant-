# Company Policy RAG Assistant

An intelligent question-answering system for employees built using Retrieval-Augmented Generation (RAG).  
The assistant retrieves relevant information from company policy documents and generates natural language answers using a language model.

---

## 🚩 Problem

Employees often struggle to locate accurate information inside long policy documents such as leave rules, reimbursement processes, workplace conduct, etc.

Traditional keyword search is slow and requires manual reading.

This project builds an AI assistant that:
- understands user queries,
- retrieves the most relevant policy passages,
- and produces clear answers in natural language.

---

## 🧠 Architecture

Pipeline:

1. User asks a question
2. DPR Context Encoder converts documents into embeddings
3. Similar passages are retrieved via vector search
4. Retrieved context is passed to GPT-2
5. GPT-2 generates the final answer grounded in company policies

---

## 🛠 Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- DPR (Dense Passage Retrieval)
- GPT-2
- FAISS / vector similarity search (if used)

---

## 📂 Project Structure

