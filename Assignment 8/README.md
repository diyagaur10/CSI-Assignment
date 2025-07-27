# 💬 RAG Q&A Chatbot for Loan Applications

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on a loan approval dataset. It uses document retrieval (FAISS) and generative AI (GPT-based models) to provide context-aware, intelligent responses.

---

## 🎯 Objective

To build a chatbot that can:
- Understand natural language queries about loan data
- Retrieve the most relevant entries from the dataset
- Generate intelligent answers using a lightweight generative model

---

## 📊 Dataset

**Source**: [Loan Approval Dataset – Kaggle](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction)

Contains demographic and financial information of loan applicants, including income, loan amount, credit history, and approval status.

---

## 🔧 Tech Stack

- Python
- HuggingFace Transformers
- FAISS (Facebook AI Similarity Search)
- PyTorch
- scikit-learn
- Pandas, NumPy

---

## 📂 Project Structure


---

## 🛠️ How It Works

1. **Load Dataset**  
   Converts CSV rows into descriptive text documents.

2. **Embed Documents**  
   Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings.

3. **Build FAISS Index**  
   Vector index enables fast semantic search.

4. **Document Retrieval**  
   Retrieves top-k documents relevant to the user query.

5. **Response Generation**  
   Uses `DistilGPT2` (or fallback to GPT-2) to generate contextual answers.

6. **Interactive Chat**  
   Supports real-time chat with users and fallback rule-based answers.

---

## 💬 Example Queries

- "What is the average income of loan applicants?"
- "How many loans were approved vs rejected?"
- "What is the loan amount range in the dataset?"
- "How does credit history affect loan approval?"
- "Tell me about applicants with high income."

---

## ▶️ Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd RAG-LoanChatbot
