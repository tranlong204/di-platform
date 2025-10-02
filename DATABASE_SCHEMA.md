# Database Schema Documentation

## 📊 Complete Database Tables Overview

### 🗄️ **All 5 Tables in PostgreSQL:**

| Table Name | Size | Records | Purpose |
|------------|------|---------|---------|
| **documents** | 16 kB | 2 | Document metadata & file info |
| **document_chunks** | 80 kB | 3 | Text chunks for vector search |
| **queries** | 64 kB | 10+ | User queries & AI responses |
| **agents** | 8 kB | 0 | AI agent configurations |
| **agent_executions** | 8 kB | 0 | Agent execution logs |

---

## 📄 **1. DOCUMENTS TABLE**
**Purpose:** Stores uploaded document metadata

**Columns:**
- `id` (PK) - Unique document ID
- `filename` - Original filename
- `file_path` - File location on disk
- `file_type` - File extension (pdf, txt, etc.)
- `file_size` - File size in bytes
- `content_hash` - SHA-256 hash for deduplication
- `title` - Document title (optional)
- `summary` - Document summary (optional)
- `metadata_json` - Additional metadata
- `processed` - Processing status flag
- `created_at` / `updated_at` - Timestamps

**Current Data (2 documents):**
- Java_Stream_25_questions.pdf (14.9 MB)
- System_design_10_core_concepts.pdf (3.5 MB)

---

## 🧩 **2. DOCUMENT_CHUNKS TABLE**
**Purpose:** Stores text chunks extracted from documents

**Columns:**
- `id` (PK) - Unique chunk ID
- `document_id` (FK) - Reference to documents table
- `chunk_index` - Order within document
- `content` - Actual text content
- `content_hash` - Hash of chunk content
- `token_count` - Number of tokens in chunk
- `embedding_id` - Reference to FAISS vector index
- `metadata_json` - Chunk metadata
- `created_at` - Creation timestamp

**Current Data (3 chunks):**
- Document 6: 2 chunks (1000 + 540 tokens)
- Document 7: 1 chunk (471 tokens)

---

## ❓ **3. QUERIES TABLE**
**Purpose:** Stores user queries and AI responses

**Columns:**
- `id` (PK) - Unique query ID
- `query_text` - User's question
- `response_text` - AI's response
- `query_hash` - Hash for caching
- `context_documents` - JSON array of document IDs used
- `context_chunks` - JSON array of chunk IDs used
- `similarity_scores` - JSON array of similarity scores
- `processing_time` - Time taken to process (seconds)
- `user_feedback` - User rating (1-5)
- `created_at` - Query timestamp

**Current Data (10+ queries):**
- Recent queries about Java Streams, Distributed Cache, System Design
- Processing times range from 0.001s to 18.99s

---

## 🤖 **4. AGENTS TABLE**
**Purpose:** Stores AI agent configurations

**Columns:**
- `id` (PK) - Unique agent ID
- `name` - Agent name
- `description` - Agent description
- `workflow_config` - JSON workflow configuration
- `tools` - JSON array of available tools
- `active` - Whether agent is active
- `created_at` / `updated_at` - Timestamps

**Current Data:** Empty (0 agents configured)

---

## ⚡ **5. AGENT_EXECUTIONS TABLE**
**Purpose:** Tracks agent execution logs

**Columns:**
- `id` (PK) - Unique execution ID
- `agent_id` (FK) - Reference to agents table
- `query_id` (FK) - Reference to queries table
- `execution_steps` - JSON array of execution steps
- `final_result` - Final execution result
- `execution_time` - Time taken to execute
- `success` - Whether execution succeeded
- `error_message` - Error details if failed
- `created_at` - Execution timestamp

**Current Data:** Empty (0 executions recorded)

---

## 📈 **Database Summary:**
- **Total Size:** ~176 kB
- **Active Data:** Documents, chunks, and queries
- **Empty Tables:** Agents and agent executions
- **Extensions:** pgvector enabled for vector operations
- **Indexes:** Optimized for fast lookups on IDs and hashes

## 🔗 **Table Relationships:**
```
documents (1) ──→ (many) document_chunks
queries (1) ──→ (many) agent_executions
agents (1) ──→ (many) agent_executions
```

## 🛠️ **Database Configuration:**
- **Database:** di_platform
- **User:** longtran
- **Host:** localhost:5432
- **Extensions:** pgvector
- **Engine:** PostgreSQL 15

---

*Last Updated: October 1, 2025*
*Generated from live database inspection*
