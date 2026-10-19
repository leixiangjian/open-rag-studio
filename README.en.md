# OpenRAG Studio

**OpenRAG Studio** is a locally deployed, enterprise-grade Retrieval-Augmented Generation (RAG) knowledge base platform. It deeply integrates local large language model services (OpenCode) and vector databases (ChromaDB), helping users easily transform private documents into intelligent assets for secure, accurate, and context-aware Q&A.

## 📖 Detailed Business Features

OpenRAG Studio consists of four core business modules, covering the complete lifecycle from model diagnostics and knowledge ingestion to intelligent Q&A.

### 1. Configuration Center
The Configuration Center is the foundation of the system, providing out-of-the-box access to underlying services.
- **OpenCode Service & Model Management**: Automatically detects the connectivity of the local OpenCode service. Upon successful connection, the system automatically fetches the **list of free LLMs** provided by the server. Users can easily select and apply the desired model from a dropdown menu, with configurations synchronized to the backend in real-time.
- **Diagnostic Console**: A built-in lightweight chat window. Before officially using the knowledge base, users can test message sending and receiving directly with the LLM to verify model inference and network connectivity.
- **Vector Database Probing**: Automatically performs heartbeat checks on the local ChromaDB during system startup and operations, displaying the connection status in real-time to ensure a smooth data storage and retrieval pipeline.

### 2. Document Ingestion & Management
Efficiently transforms unstructured documents into vector data understandable by LLMs, with support for subsequent data governance.
- **Multi-format Document Parsing**: Supports drag-and-drop upload of PDF, Word (DOCX), and TXT files (max 20MB per file).
- **Visual Chunking Configuration**: Provides a wizard-based document chunking feature. Users can customize `Chunk Size` and `Chunk Overlap`, and preview the chunking results in real-time to ensure optimal retrieval accuracy.
- **One-click Vectorization & Ingestion**: After confirming the chunks, the system automatically calls the Embedding model to vectorize them and writes them into the vector database, with visual progress tracking throughout the process.
- **Knowledge Base Asset Dashboard**: Provides a list management for ingested documents. Users can view the specific chunk details of any document at any time and support **one-click complete deletion** of documents and all their associated vector data, ensuring the purity and timeliness of the knowledge base.

### 3. Multi-session Intelligent Q&A
The core interactive interface based on RAG technology, offering a smooth ChatGPT-like experience.
- **Multi-session Isolation & Persistence**: Supports creating, switching, and deleting multiple independent chat windows. All conversation histories are automatically saved locally in the browser (LocalStorage) and persist across page refreshes or closures, making it easy for users to manage conversations by different business topics.
- **Contextual Memory**: The system automatically maintains the context of the current session, carrying the recent history with each query. This allows the LLM to understand follow-up intents and achieve coherent multi-turn conversations.
- **Streaming Typewriter Output**: Answers are output character-by-character in a streaming manner, providing an excellent reading experience. It is accompanied by breathing-light prompts like "Retrieving knowledge base" and "Thinking and generating" to eliminate waiting anxiety.
- **Retrieval-Augmented Generation (RAG)**: The system automatically vectorizes user questions and recalls the most relevant chunks from the knowledge base. These are concatenated into the Prompt as reference materials, fundamentally reducing LLM hallucinations and resulting in more accurate answers.

### 4. Built-in Developer Manual
- **Zero-threshold Onboarding**: The system includes a detailed developer and operations manual, covering environment preparation, dependency installation, backend startup, configuration guides, and FAQ, helping team members quickly take over and maintain the project.