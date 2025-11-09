Need of File and Folder in Server

__pycache__ → Auto-generated folder storing compiled Python bytecode for faster execution.

.venv → Virtual environment for server dependencies.

auth/ → Handles authentication:

hash_utils.py → Functions to hash passwords or verify them.

models.py → Database models for users or auth data.

routes.py → API endpoints for login, signup, etc.

__pycache__/ → Bytecode for auth module.

chat/ → Likely handles chat logic or LLM interactions. Contains its own __pycache__/.

config/ → Configuration files (DB connections, API keys).

uploaded_docs/ → Stores documents uploaded by users for RAG.

.env → Stores environment variables (API keys, DB URIs).

.python-version → Specifies Python version.

main.py → Entry point for server (starts API or backend logic).

pyproject.toml → Project metadata & dependencies.

README.md → Server instructions/documentation.

requirements.txt → List of Python packages needed for server.



Server Workflow with Supporting Files:


Start Server: main.py → loads app and initializes backend.

Load Config: .env + config/ → API keys, DB URIs, Pinecone settings.

Authentication: auth/routes.py → handles login/signup, uses hash_utils.py and models.py.

Document Upload: uploaded_docs/ → stores user documents.

Chunking & Embedding: Server code (likely in chat/) splits docs, creates embeddings, stores in Pinecone or DB.

Query Handling: API receives query → converts to embedding → retrieves relevant chunks.

LLM Generation: Gemini API called to generate answer from retrieved chunks.

Send Response: Server sends result back to client.

Chat Management: chat/ → handles conversation context/history if needed.

Supporting Files:

__pycache__/ → compiled Python bytecode.

.venv → server’s virtual environment.

requirements.txt → install dependencies.

pyproject.toml → project metadata and packages.

.python-version → specifies Python version.

README.md → server documentation.