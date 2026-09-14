# 📚 Machine Learning Lecture RAG Assistant

A Retrieval-Augmented Generation (RAG) system built with **LangChain, FAISS, Hugging Face embeddings, and Google Gemini** to answer questions from Andrew Ng's Machine Learning lecture material.

The project demonstrates the complete RAG pipeline — from document loading and chunking to embedding generation, vector similarity search, context retrieval, prompt construction, and LLM-based answer generation.

---

## 🚀 Project Overview

Large Language Models can generate impressive answers, but they may not have access to a specific private or domain-specific knowledge base.

This project addresses that problem by building a **Retrieval-Augmented Generation pipeline** over Machine Learning lecture PDFs.

Instead of asking the LLM to answer directly, the system:

1. Accepts a user's question.
2. Converts the question into an embedding.
3. Searches a vector store for semantically relevant lecture chunks.
4. Retrieves the most relevant pieces of information.
5. Injects the retrieved information into a prompt.
6. Sends the grounded prompt to Google Gemini.
7. Generates an answer based on the retrieved lecture content.

### High-Level Architecture

```text
                  User Question
                       │
                       ▼
                Query Embedding
                       │
                       ▼
              ┌─────────────────┐
              │   FAISS Vector  │
              │      Store      │
              └─────────────────┘
                       │
                Similarity Search
                       │
                       ▼
              Top-K Relevant Chunks
                       │
                       ▼
                 Prompt Template
                       │
                       ▼
                 Google Gemini
                       │
                       ▼
                  Final Answer
```

---

# 🎯 Objectives

The main objectives of this project are:

- Understand the fundamentals of Retrieval-Augmented Generation.
- Build a RAG pipeline from scratch using LangChain.
- Process PDF-based knowledge sources.
- Split long documents into smaller retrievable chunks.
- Generate semantic embeddings locally.
- Store and retrieve embeddings using FAISS.
- Perform semantic similarity search.
- Experiment with different retrieval strategies.
- Understand similarity scores and distance metrics.
- Explore MMR-based retrieval.
- Use metadata filtering for more controlled retrieval.
- Ground LLM responses in retrieved context.
- Handle questions outside the available knowledge base.

---

# 📂 Knowledge Base

The current knowledge base consists of Andrew Ng's Machine Learning lecture material covering:

### Lecture 01

Topics include:

- Introduction to Machine Learning
- Supervised Learning
- Unsupervised Learning
- Regression
- Classification
- Clustering
- Machine Learning examples and applications

### Lecture 02

Topics include:

- Linear Regression
- Hypothesis Functions
- Parameters and Features
- Cost Function
- Least Squares
- Gradient Descent
- Learning Rate
- Convergence
- Normal Equation

### Lecture 03

Topics include:

- Feature Selection
- Parametric vs Non-Parametric Learning
- Locally Weighted Regression
- Probabilistic Interpretation of Linear Regression
- Logistic Regression
- Sigmoid Function
- Probabilistic Classification
- Perceptron
- Newton's Method

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| LangChain | RAG pipeline orchestration |
| Hugging Face / Sentence Transformers | Local text embeddings |
| `all-MiniLM-L6-v2` | Embedding model |
| FAISS | Vector similarity search |
| Google Gemini | Large Language Model |
| Google Colab | Development environment |
| PyPDFLoader | PDF document loading |

---

# 🔄 RAG Pipeline

## 1. Document Loading

The lecture PDFs are loaded using LangChain's PDF loader.

Each PDF is converted into LangChain `Document` objects containing:

- `page_content`
- `metadata`

Example metadata can include:

```text
source → lecture PDF
page   → page number
```

---

## 2. Text Splitting

Large documents are split into smaller chunks using LangChain text splitters.

The project uses:

- `chunk_size`
- `chunk_overlap`

### Why chunking?

Embedding an entire lecture as one vector would make retrieval too coarse.

Instead:

```text
Lecture PDF
     ↓
Pages
     ↓
Smaller Chunks
     ↓
Embeddings
```

Smaller chunks allow the retriever to find more specific pieces of information.

### Chunk Overlap

A small overlap is maintained between consecutive chunks so that important information near chunk boundaries is less likely to be lost.

---

# 🧠 3. Embeddings

Each text chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model converts text into a **384-dimensional vector representation**.

Conceptually:

```text
"Supervised learning learns from labeled examples"
                         │
                         ▼
                Embedding Model
                         │
                         ▼
              [0.12, -0.04, ..., 0.27]
```

The resulting vectors represent semantic information about the text.

---

# 🗄️ 4. FAISS Vector Store

The generated embeddings are stored in a FAISS vector index.

FAISS enables efficient vector similarity search.

The architecture becomes:

```text
Document Chunk
      ↓
Embedding
      ↓
384-D Vector
      ↓
FAISS Index
```

When a user asks a question, the question is also converted into an embedding and compared against the stored vectors.

---

# 🔎 5. Similarity Search

The user's query is converted into an embedding and searched against the FAISS index.

For example:

```text
User:
"What is regression?"
```

The system retrieves the most semantically relevant lecture chunks.

The current implementation uses:

```text
Top-K Retrieval
```

where `k` determines how many documents are returned.

For example:

```text
k = 3
```

means the system retrieves the top 3 relevant chunks.

---

# 📊 Similarity Scores and Distance

The project also explores retrieval scores to understand how relevant retrieved documents are to a query.

In the current FAISS configuration, raw scores correspond to **L2 / Euclidean distance**.

Important distinction:

```text
Lower distance → More similar
Higher distance → Less similar
```

LangChain can also expose transformed relevance scores.

These scores should not automatically be interpreted as percentages of semantic similarity.

Their interpretation depends on factors such as:

- Embedding model
- Distance metric
- Vector normalization
- Vector store configuration

---

# 🔀 6. Maximal Marginal Relevance (MMR)

The project also explores **Maximal Marginal Relevance (MMR)** retrieval.

A basic similarity search can sometimes return multiple chunks containing almost identical information.

MMR attempts to balance:

```text
Relevance to Query
        +
Diversity of Retrieved Documents
```

Conceptually:

```text
Query
  ↓
Candidate Documents
  ↓
MMR
  ↓
Relevant + Less Redundant Documents
```

The main parameters explored are:

- `k` — number of final documents
- `fetch_k` — number of candidate documents considered
- `lambda_mult` — relevance/diversity trade-off

MMR is particularly useful when the retrieved top-k results are highly redundant.

---

# 🏷️ 7. Metadata Filtering

The project also explores metadata-based filtering.

For example, instead of searching across all lectures:

```text
Question
   ↓
Filter → Lecture 03
   ↓
Similarity Search
   ↓
Relevant Lecture 03 Chunks
```

Metadata filtering can be useful when the user specifies a particular:

- Lecture
- Document
- Page
- Category
- Source

Metadata filtering and semantic similarity serve different purposes:

```text
Metadata Filter
"What subset should I search?"

Similarity Search
"Which chunks inside that subset are relevant?"
```

---

# 🤖 8. Prompt Construction

After retrieval, the relevant chunks are passed to the LLM as context.

The prompt contains:

```text
Question
+
Retrieved Context
+
Instructions
```

Conceptually:

```text
              Retrieved Context
                     │
                     ▼
User Question → Prompt Template
                     │
                     ▼
                 Gemini LLM
                     │
                     ▼
                Final Answer
```

A grounding instruction is also used so that the model does not rely on unsupported information.

If the answer cannot be found in the retrieved context, the system can respond with:

```text
I DON'T KNOW
```

---

# 🧠 9. Google Gemini

Google Gemini is used as the generative LLM.

Its role is **not to perform the initial document retrieval**.

Instead:

```text
FAISS
   ↓
Find relevant information

Gemini
   ↓
Understand retrieved information
and generate the answer
```

This separation between retrieval and generation is a fundamental part of RAG.

---

# 🔄 Complete End-to-End Flow

```text
                  ┌──────────────┐
                  │ Lecture PDFs │
                  └──────┬───────┘
                         │
                         ▼
                  PDF Document Loader
                         │
                         ▼
                    Text Splitting
                         │
                         ▼
                     Text Chunks
                         │
                         ▼
                  Embedding Model
                         │
                         ▼
                  384-Dimensional
                      Vectors
                         │
                         ▼
                    FAISS Index
                         │
                         │
User Question ───────────┘
      │
      ▼
Query Embedding
      │
      ▼
Similarity Search / MMR
      │
      ▼
Relevant Lecture Chunks
      │
      ▼
Prompt Template
      │
      ▼
Google Gemini
      │
      ▼
Grounded Answer
```

---

# 🧪 Retrieval Experiments

Several types of questions were used to evaluate the behavior of the RAG pipeline.

### Example 1 — In-domain Question

```text
Question:
"What is regression?"
```

The system retrieves relevant lecture material and generates an answer describing regression as a supervised learning problem involving prediction of continuous values.

---

### Example 2 — Out-of-domain Question

```text
Question:
"What is an LLM?"
```

Since the current knowledge base focuses on Andrew Ng's Machine Learning lectures, the system should not rely on its general knowledge to answer the question.

Expected behavior:

```text
I DON'T KNOW
```

This tests whether the RAG system respects its available context.

---

### Example 3 — Conceptual Retrieval

```text
Question:
"How does a learning algorithm learn from examples
where the correct answers are provided?"
```

The system retrieves relevant material related to supervised learning and generates an answer based on the retrieved context.

---

### Example 4 — Logistic Regression

```text
Question:
"How does logistic regression use the sigmoid function
to perform binary classification?"
```

The system retrieves lecture content discussing:

- Logistic regression
- Sigmoid function
- Probability interpretation
- Binary classification

---

# ⚠️ Retrieval Evaluation Insight

One important observation from experimentation is that **similarity scores alone should not be used to define a universal retrieval threshold**.

For example, a query can receive relatively high similarity scores while the retrieved context may still be insufficient to fully answer a complex question.

Therefore:

```text
High Similarity Score
        ≠
Guaranteed Answerability
```

Retrieval quality should be evaluated using multiple factors:

- Relevance of retrieved chunks
- Context sufficiency
- Answer correctness
- Groundedness
- Out-of-domain behavior

This is an important consideration when designing reliable RAG systems.

---

# 🔐 API Key Security

API credentials are **not hard-coded inside the notebook**.

Google API credentials are stored using **Google Colab Secrets**.

The repository should never contain:

```text
API keys
Passwords
Tokens
Credentials
```

If this project is made public, credentials must remain outside the source code.

---

# 📁 Project Structure

A possible repository structure:

```text
machine-learning-rag/
│
├── README.md
│
├── notebooks/
│   └── machine_learning_rag.ipynb
│
├── data/
│   └── README.md
│
├── requirements.txt
│
└── .gitignore
```

The lecture PDFs themselves may be excluded from the repository if redistribution rights do not permit including them.

---

# ▶️ How to Run

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd machine-learning-rag
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure Google Gemini

Create a Google Gemini API key and store it securely using the environment/secret mechanism used by your notebook.

Do **not** hard-code the key in the notebook.

## 4. Open the Notebook

The project can be run using Google Colab or Jupyter Notebook.

## 5. Load the Lecture Documents

Place the required lecture PDFs in the expected data location.

## 6. Run the RAG Pipeline

Execute the notebook cells sequentially.

The final system will accept a user query and retrieve relevant lecture content before generating an answer.

---

# 📦 Dependencies

The project uses packages from the LangChain ecosystem along with FAISS and Sentence Transformers.

Typical dependencies include:

```text
langchain
langchain-core
langchain-community
langchain-text-splitters
langchain-google-genai
langchain-huggingface
sentence-transformers
faiss-cpu
pypdf
```

Exact package versions should be pinned in `requirements.txt` for reproducibility.

---

# 🧩 Key Concepts Demonstrated

This project provides hands-on implementation of several important RAG concepts:

- Retrieval-Augmented Generation
- Document Loading
- Text Chunking
- Chunk Overlap
- Embeddings
- Vector Representations
- Vector Stores
- FAISS
- Semantic Search
- Top-K Retrieval
- Similarity Search
- L2 / Euclidean Distance
- Relevance Scores
- MMR
- Metadata
- Metadata Filtering
- Prompt Templates
- Context Grounding
- Hallucination Control
- Out-of-domain Detection

---

# 🚧 Current Limitations

This is currently a **learning/prototype implementation** rather than a production-ready RAG application.

Current limitations include:

- Limited knowledge base
- Fixed chunking strategy
- Basic retrieval configuration
- No dedicated retrieval evaluation framework
- No reranking stage
- No hybrid search
- No production vector database
- No API/backend layer
- No frontend application
- No automated evaluation pipeline
- No monitoring/observability
- No authentication or user-level access control

---

# 🔮 Future Improvements

The project will be extended progressively toward a more robust and production-oriented RAG system.

### Advanced Retrieval

- Better chunking strategies
- Metadata-aware retrieval
- Hybrid search
- Sparse + dense retrieval
- Reranking
- Query rewriting
- Query decomposition
- Multi-query retrieval
- Contextual compression
- Retrieval evaluation

### Agentic RAG

The next major stage is to convert the existing RAG pipeline into a tool that can be used by an AI agent.

Potential architecture:

```text
                         User
                          │
                          ▼
                        Agent
                     ↙    │    ↘
                   RAG   Web  Calculator
                  Tool   Tool    Tool
                   │
                 FAISS
                   │
             Lecture Knowledge
                     │
                     ▼
                    LLM
                     │
                     ▼
               Final Response
```

This would allow the system to decide which tool is appropriate for different types of questions.

---

# 🏗️ Planned Production Architecture

The long-term goal is to evolve the notebook prototype into a complete application:

```text
                User Interface
                      │
                      ▼
                   FastAPI
                      │
                      ▼
                 Agent Layer
                ↙    ↓     ↘
             RAG    Web   Calculator
              │
              ▼
        Vector Database
              │
              ▼
       Retrieval Pipeline
              │
              ▼
             LLM
              │
              ▼
       Grounded Response
```

Potential future additions:

- FastAPI
- Streamlit
- Docker
- Production vector database
- Automated testing
- Retrieval evaluation
- LLM evaluation
- Logging
- Monitoring
- Observability
- CI/CD

---

# 📈 Learning Progression

This project is being developed progressively rather than treating RAG as a single API call.

```text
Basic RAG
   │
   ├── Document Processing
   ├── Chunking
   ├── Embeddings
   ├── Vector Search
   ├── Prompting
   └── LLM Generation
          │
          ▼
Advanced Retrieval
   │
   ├── MMR
   ├── Metadata Filtering
   ├── Self-Query
   ├── Query Transformation
   ├── Reranking
   └── Contextual Compression
          │
          ▼
Agentic RAG
   │
   ├── Tool Calling
   ├── RAG as a Tool
   ├── Web Search
   ├── Calculator
   └── Agentic Decision Making
          │
          ▼
Production RAG System
   │
   ├── API
   ├── UI
   ├── Evaluation
   ├── Monitoring
   └── Deployment
```

---

# 📚 Learning Resources

This project was developed while studying Retrieval-Augmented Generation and LangChain concepts through hands-on implementation and experimentation.

Key technologies:

- LangChain
- FAISS
- Hugging Face Sentence Transformers
- Google Gemini
- Retrieval-Augmented Generation

---

# 👨‍💻 Author

**Garvit Babbar**

This project is part of my hands-on learning journey in:

- Data Science
- Machine Learning
- Generative AI
- Large Language Models
- Retrieval-Augmented Generation
- Agentic AI

The goal is to progressively transform a basic RAG prototype into a reliable, evaluated, and production-oriented AI system.

---

# ⭐ Project Status

**Current Status:** 🟢 Basic RAG Pipeline Functional

Implemented:

- ✅ PDF ingestion
- ✅ Text splitting
- ✅ Local embeddings
- ✅ FAISS vector store
- ✅ Dynamic query retrieval
- ✅ Similarity search
- ✅ Prompt templating
- ✅ Gemini generation
- ✅ MMR experimentation
- ✅ Metadata filtering
- ✅ Retrieval experiments

In Progress:

- 🔄 Advanced retrieval evaluation
- 🔄 Contextual compression
- 🔄 Agentic RAG

Planned:

- ⏳ Tool calling
- ⏳ Web search integration
- ⏳ RAG Agent
- ⏳ Evaluation framework
- ⏳ FastAPI
- ⏳ Deployment
