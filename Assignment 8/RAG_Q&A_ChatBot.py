Objective : RAG Q&A chatbot using document retrieval and generative AI for intelligent response generation (can use any light model from hugging face or a license llm(opneai, claude, grok, gemini) if free credits available
Resources : https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction?select=Training+Dataset.csv

CODE:
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, 
    pipeline, BitsAndBytesConfig
)
import faiss
import torch
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class RAGChatbot:
    def __init__(self, csv_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize RAG Chatbot
        
        Args:
            csv_path: Path to the loan dataset CSV
            embedding_model: Hugging Face model for embeddings
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load and process data
        self.df = pd.read_csv(csv_path)
        self.processed_docs = []
        self.embeddings = None
        self.index = None
        
        # Initialize models
        self.embedding_model = embedding_model
        self.tokenizer = None
        self.model = None
        self.generator = None
        
        self._setup_models()
        self._process_documents()
        self._create_embeddings()
        self._setup_faiss_index()
        
    def _setup_models(self):
        """Setup embedding and generation models"""
        print("Loading embedding model...")
        
        # For embeddings - using sentence-transformers compatible model
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        self.model = AutoModel.from_pretrained(self.embedding_model)
        self.model.to(self.device)
        
        print("Loading generation model...")
        # Using a lightweight generation model
        generation_model = "microsoft/DialoGPT-medium"
        
        try:
            self.generator = pipeline(
                "text-generation",
                model="distilgpt2",  # Lightweight alternative
                tokenizer="distilgpt2",
                device=0 if torch.cuda.is_available() else -1,
                max_length=200,
                pad_token_id=50256  # Set pad token
            )
        except Exception as e:
            print(f"Error loading generation model: {e}")
            # Fallback to basic model
            self.generator = pipeline(
                "text-generation",
                model="gpt2",
                device=0 if torch.cuda.is_available() else -1,
                max_length=150
            )
    
    def _process_documents(self):
        """Convert dataset rows into text documents"""
        print("Processing documents...")
        
        # Create descriptive text for each loan application
        for idx, row in self.df.iterrows():
            doc_text = f"""
            Loan Application ID: {idx}
            Gender: {row.get('Gender', 'Not specified')}
            Married: {row.get('Married', 'Not specified')}
            Dependents: {row.get('Dependents', 'Not specified')}
            Education: {row.get('Education', 'Not specified')}
            Self Employed: {row.get('Self_Employed', 'Not specified')}
            Applicant Income: ${row.get('ApplicantIncome', 0):,}
            Coapplicant Income: ${row.get('CoapplicantIncome', 0):,}
            Loan Amount: ${row.get('LoanAmount', 0) * 1000:,} (in thousands)
            Loan Amount Term: {row.get('Loan_Amount_Term', 'Not specified')} months
            Credit History: {row.get('Credit_History', 'Not specified')}
            Property Area: {row.get('Property_Area', 'Not specified')}
            Loan Status: {row.get('Loan_Status', 'Not specified')}
            """
            self.processed_docs.append(doc_text.strip())
    
    def _get_embeddings(self, texts):
        """Generate embeddings for text using sentence transformers approach"""
        embeddings = []
        
        for text in texts:
            # Tokenize and encode
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def _create_embeddings(self):
        """Create embeddings for all documents"""
        print("Creating embeddings...")
        self.embeddings = self._get_embeddings(self.processed_docs)
        print(f"Created embeddings shape: {self.embeddings.shape}")
    
    def _setup_faiss_index(self):
        """Setup FAISS index for similarity search"""
        print("Setting up FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        print(f"FAISS index created with {self.index.ntotal} documents")
    
    def retrieve_documents(self, query, k=3):
        """Retrieve most relevant documents for a query"""
        # Get query embedding
        query_embedding = self._get_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        retrieved_docs = []
        for i, idx in enumerate(indices[0]):
            retrieved_docs.append({
                'content': self.processed_docs[idx],
                'score': scores[0][i],
                'index': idx
            })
        
        return retrieved_docs
    
    def generate_response(self, query, retrieved_docs):
        """Generate response using retrieved documents"""
        # Create context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in retrieved_docs[:2]])  # Use top 2 docs
        
        # Create prompt
        prompt = f"""Based on the following loan application data, please answer the question:

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            # Generate response
            response = self.generator(
                prompt,
                max_length=len(prompt.split()) + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            # Extract only the answer part
            answer = generated_text.split("Answer:")[-1].strip()
            
            return answer
            
        except Exception as e:
            print(f"Generation error: {e}")
            # Fallback to rule-based response
            return self._fallback_response(query, retrieved_docs)
    
    def _fallback_response(self, query, retrieved_docs):
        """Fallback response generation using simple rules"""
        query_lower = query.lower()
        
        # Extract key information from retrieved docs
        if retrieved_docs:
            doc_content = retrieved_docs[0]['content']
            
            if 'income' in query_lower:
                if 'Applicant Income' in doc_content:
                    lines = doc_content.split('\n')
                    for line in lines:
                        if 'Applicant Income' in line or 'Coapplicant Income' in line:
                            return f"Based on the data: {line.strip()}"
            
            elif 'loan amount' in query_lower:
                lines = doc_content.split('\n')
                for line in lines:
                    if 'Loan Amount:' in line and 'Term' not in line:
                        return f"Based on the data: {line.strip()}"
            
            elif 'status' in query_lower or 'approved' in query_lower:
                lines = doc_content.split('\n')
                for line in lines:
                    if 'Loan Status' in line:
                        return f"Based on the data: {line.strip()}"
            
            elif 'credit' in query_lower:
                lines = doc_content.split('\n')
                for line in lines:
                    if 'Credit History' in line:
                        return f"Based on the data: {line.strip()}"
        
        return "I found relevant loan application data, but I need a more specific question to provide a detailed answer."
    
    def chat(self, query):
        """Main chat function"""
        print(f"\nQuery: {query}")
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, k=3)
        print(f"Retrieved {len(retrieved_docs)} documents")
        
        # Generate response
        response = self.generate_response(query, retrieved_docs)
        
        return {
            'query': query,
            'response': response,
            'retrieved_docs': retrieved_docs,
            'sources': [f"Loan Application ID: {doc['index']}" for doc in retrieved_docs]
        }

# Dataset Analysis Functions
def analyze_dataset(df):
    """Analyze the loan dataset"""
    print("=== DATASET ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    if 'Loan_Status' in df.columns:
        print(f"\nLoan Status distribution:")
        print(df['Loan_Status'].value_counts())
    
    print(f"\nFirst few rows:")
    print(df.head())

# Example usage and demo
def run_demo():
    """Run demonstration of the RAG chatbot"""
    
    # Note: Replace with actual path to your dataset
    csv_path = "loan_dataset.csv"  # Update this path
    
    try:
        # Initialize chatbot
        print("Initializing RAG Chatbot...")
        chatbot = RAGChatbot(csv_path)
        
        # Analyze dataset first
        analyze_dataset(chatbot.df)
        
        # Demo queries
        demo_queries = [
            "What is the average income of loan applicants?",
            "How many loans were approved vs rejected?",
            "What factors affect loan approval?",
            "Tell me about applicants with high income",
            "What is the loan amount range in the dataset?",
            "How does credit history affect loan approval?"
        ]
        
        print("\n" + "="*50)
        print("RAG CHATBOT DEMO")
        print("="*50)
        
        for query in demo_queries:
            result = chatbot.chat(query)
            print(f"\nQ: {result['query']}")
            print(f"A: {result['response']}")
            print(f"Sources: {', '.join(result['sources'][:2])}")
            print("-" * 40)
        
        # Interactive mode
        print("\n" + "="*50)
        print("INTERACTIVE MODE (type 'quit' to exit)")
        print("="*50)
        
        while True:
            user_query = input("\nYour question: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_query:
                result = chatbot.chat(user_query)
                print(f"\nAnswer: {result['response']}")
                print(f"Sources: {', '.join(result['sources'][:2])}")
    
    except FileNotFoundError:
        print(f"Error: Could not find dataset file at {csv_path}")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction")
        print("And update the csv_path variable with the correct path.")
    
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        print("Make sure you have installed required packages:")
        print("pip install pandas numpy transformers faiss-cpu torch scikit-learn")

if __name__ == "__main__":
    run_demo()

# Additional utility functions
def create_sample_dataset():
    """Create a sample dataset if the original is not available"""
    sample_data = {
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Married': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'Dependents': ['0', '1', '2', '0', '3+'],
        'Education': ['Graduate', 'Graduate', 'Not Graduate', 'Graduate', 'Graduate'],
        'Self_Employed': ['No', 'No', 'Yes', 'No', 'Yes'],
        'ApplicantIncome': [5849, 4583, 3000, 2583, 6000],
        'CoapplicantIncome': [0, 1508, 0, 2358, 0],
        'LoanAmount': [128, 128, 66, 120, 141],
        'Loan_Amount_Term': [360, 360, 360, 360, 360],
        'Credit_History': [1, 1, 1, 1, 1],
        'Property_Area': ['Urban', 'Rural', 'Urban', 'Urban', 'Urban'],
        'Loan_Status': ['Y', 'N', 'Y', 'Y', 'Y']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_loan_dataset.csv', index=False)
    print("Sample dataset created as 'sample_loan_dataset.csv'")
    return df

# Installation requirements
requirements = """
Required packages:
pip install pandas numpy transformers faiss-cpu torch scikit-learn

For GPU support (optional):
pip install faiss-gpu

For better sentence transformers:
pip install sentence-transformers
"""

print(requirements)
