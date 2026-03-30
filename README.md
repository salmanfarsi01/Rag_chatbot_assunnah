# As-Sunnah Foundation RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built for the **As-Sunnah Foundation** knowledge base. It lets you ask questions in **text or voice**, retrieves relevant context from a Pinecone vector database, and generates grounded answers using the **Qwen 2.5** language model — without hallucinating outside the provided documents.

---

<img width="1769" height="828" alt="image" src="https://github.com/user-attachments/assets/f9bd5387-8af7-4e78-ba14-d00eb2c1c597" />

<img width="1790" height="611" alt="image" src="https://github.com/user-attachments/assets/a872b7fb-c1d7-45c5-be74-ad3572967ea0" />



## Project Structure

```
as-sunnah-chatbot/
│
├── C_Rag_Embed.ipynb       # Document ingestion pipeline (embed & upload to Pinecone)
├── C_Rag_AI.ipynb          # RAG inference pipeline (query, retrieve, generate)
└── README.md
```

---

## How It Works

```
Your Documents (TXT / PDF / DOCX)
        │
        ▼
  [C_Rag_Embed.ipynb]
  Load → Chunk → Embed (BGE) → Upload to Pinecone
        │
        ▼
   Pinecone Vector DB
        │
        ▼
  [C_Rag_AI.ipynb]
  Text or Voice Query → Embed Query → Retrieve Top-K Chunks
        │
        ▼
  Build Prompt → Qwen 2.5 → Grounded Answer
```

---

## Setup

### 1. Install Dependencies

```bash
# Embedding / Ingestion pipeline
pip install pinecone sentence-transformers pypdf python-docx

# Inference pipeline
pip install pinecone sentence-transformers transformers accelerate openai-whisper
```

### 2. Set Your API Key

Replace `"my api key"` with your actual Pinecone API key in both notebooks:

```python
PINECONE_API_KEY = "your-actual-pinecone-api-key"
```

### 3. Run Ingestion First

Run `C_Rag_Embed.ipynb` to upload your documents into Pinecone. Then run `C_Rag_AI.ipynb` to start asking questions.

---

## Configuration

Both notebooks share common configuration constants at the top:

| Constant | Default | Description |
|---|---|---|
| `PINECONE_API_KEY` | `"my api key"` | Your Pinecone API key |
| `PINECONE_INDEX_NAME` | `"assunnah-index"` | Name of the Pinecone index |
| `PINECONE_CLOUD` | `"aws"` | Cloud provider for Pinecone |
| `PINECONE_REGION` | `"us-east-1"` | Region for Pinecone serverless |
| `EMBEDDING_MODEL` | `"BAAI/bge-large-en-v1.5"` | Sentence embedding model |
| `EMBEDDING_DIM` | `1024` | Embedding vector dimension |
| `CHUNK_SIZE` | `800` | Characters per text chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `QWEN_MODEL_NAME` | `"Qwen/Qwen2.5-1.5B-Instruct"` | LLM for answer generation |
| `WHISPER_MODEL_NAME` | `"base"` | Whisper model size for voice input |
| `TOP_K` | `3` | Number of chunks to retrieve per query |
| `SIMILARITY_THRESHOLD` | `0.40` | Minimum cosine similarity score to accept a result |

---

## Part 1 — Ingestion Pipeline (`C_Rag_Embed.ipynb`)

This notebook reads your documents, splits them into chunks, embeds them, and stores them in Pinecone.

---

### `read_txt(file_path)`

Reads a plain text file and returns its full content as a string.

```python
def read_txt(file_path: str) -> str
```

**Parameters:**
- `file_path` — Path to the `.txt` file

**Returns:** Full text content as a string

---

### `read_pdf(file_path)`

Reads a PDF file page by page and concatenates all extracted text.

```python
def read_pdf(file_path: str) -> str
```

**Parameters:**
- `file_path` — Path to the `.pdf` file

**Returns:** All page texts joined by newlines. Pages with no extractable text are skipped.

---

### `read_docx(file_path)`

Reads a `.docx` Word document and extracts text from every paragraph.

```python
def read_docx(file_path: str) -> str
```

**Parameters:**
- `file_path` — Path to the `.docx` file

**Returns:** All paragraph texts joined by newlines

---

### `load_document(file_path)`

A unified document loader that automatically detects the file type by extension and calls the appropriate reader (`read_txt`, `read_pdf`, or `read_docx`).

```python
def load_document(file_path: str) -> str
```

**Parameters:**
- `file_path` — Path to a `.txt`, `.pdf`, or `.docx` file

**Returns:** Full extracted text as a string

**Raises:** `ValueError` if the file extension is not supported

**Supported formats:**

| Extension | Handler |
|---|---|
| `.txt` | `read_txt()` |
| `.pdf` | `read_pdf()` |
| `.docx` | `read_docx()` |

---

### `chunk_text(text, chunk_size, overlap)`

Splits a long string into overlapping fixed-size chunks. Overlap ensures that context spanning chunk boundaries is not lost during retrieval.

```python
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]
```

**Parameters:**
- `text` — The full document text to split
- `chunk_size` — Number of characters per chunk (default: `800`)
- `overlap` — Number of characters shared between consecutive chunks (default: `150`)

**Returns:** A list of text chunks

**Example:** With `chunk_size=800` and `overlap=150`, chunk 1 covers characters 0–800, chunk 2 covers characters 650–1450, and so on.

> The function also normalizes whitespace by collapsing tabs, carriage returns, and multiple spaces before chunking.

---

### `embed_documents(texts)`

Embeds a list of text strings into dense vectors using the BGE embedding model. Vectors are L2-normalized for cosine similarity search.

```python
def embed_documents(texts: List[str]) -> List[List[float]]
```

**Parameters:**
- `texts` — A list of text strings to embed

**Returns:** A list of embedding vectors (each vector is a list of 1024 floats)

> Uses `show_progress_bar=True` so you can track progress when embedding large batches.

---

### `ingest_documents(file_paths)`

The main ingestion function. Orchestrates the full pipeline: load → chunk → embed → upload to Pinecone. Handles multiple files in one call and uploads vectors in batches of 100.

```python
def ingest_documents(file_paths: List[str]) -> None
```

**Parameters:**
- `file_paths` — A list of file paths to process

**What it does, step by step:**
1. Loops over each file path
2. Calls `load_document()` to extract text
3. Calls `chunk_text()` to split into chunks
4. Calls `embed_documents()` to generate embeddings
5. Builds a metadata dict per chunk: `{ source, chunk_id, text }`
6. Assigns each vector a unique UUID
7. Uploads all vectors to Pinecone in batches of 100

**Output:** Prints progress per file and total vectors uploaded

---

### `embed_query(query)` *(also used in AI notebook)*

Embeds a search query using the BGE model with an instruction prefix for better retrieval performance.

```python
def embed_query(query: str) -> List[float]
```

**Parameters:**
- `query` — The user's search question

**Returns:** A single normalized embedding vector (1024 floats)

> The instruction prefix `"Represent this sentence for searching relevant passages: "` is prepended to the query — this is required by the BGE model to get accurate retrieval embeddings.

---

### `search_pinecone(query, top_k)`

Embeds a query and searches Pinecone for the most similar document chunks.

```python
def search_pinecone(query: str, top_k: int = 5)
```

**Parameters:**
- `query` — The search question
- `top_k` — Number of results to return (default: `5`)

**Returns:** Pinecone query results object containing a `"matches"` list, each with `score`, `metadata.source`, `metadata.chunk_id`, and `metadata.text`

---

## Part 2 — Inference Pipeline (`C_Rag_AI.ipynb`)

This notebook handles the full question-answering flow: query → retrieve → generate → respond.

---

### `embed_query(query)`

Same function as in the ingestion notebook. Embeds the user's question with the BGE instruction prefix for accurate retrieval.

```python
def embed_query(query: str) -> List[float]
```

**Parameters:**
- `query` — The user's question as a string

**Returns:** A normalized 1024-dimensional embedding vector

---

### `retrieve_chunks(query, top_k)`

Retrieves the top-K most relevant document chunks from Pinecone for a given query.

```python
def retrieve_chunks(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]
```

**Parameters:**
- `query` — The user's question
- `top_k` — Number of chunks to retrieve (default: `TOP_K = 3`)

**Returns:** A list of Pinecone match objects. Each match contains:
- `score` — Cosine similarity score (0 to 1)
- `metadata.source` — Source filename
- `metadata.chunk_id` — Chunk index within that file
- `metadata.text` — The actual text of the chunk

---

### `format_context(matches)`

Formats the retrieved Pinecone matches into a readable context block that gets inserted into the LLM prompt.

```python
def format_context(matches: List[Dict[str, Any]]) -> str
```

**Parameters:**
- `matches` — List of Pinecone match objects from `retrieve_chunks()`

**Returns:** A formatted multi-line string like:

```
[Source 1] File: as_sunnah_foundation_extract.txt | Chunk: 19
<chunk text here>

[Source 2] File: as_sunnah_foundation_extract.txt | Chunk: 4
<chunk text here>
```

---

### `build_rag_prompt(question, context)`

Constructs the full instruction prompt that is passed to the Qwen LLM. The prompt strictly instructs the model to answer only from the retrieved context and not use outside knowledge.

```python
def build_rag_prompt(question: str, context: str) -> str
```

**Parameters:**
- `question` — The user's question
- `context` — The formatted context string from `format_context()`

**Returns:** A complete prompt string with rules, context, and the question

**Rules enforced in the prompt:**
1. Use only the retrieved context
2. Do not use outside knowledge
3. If the answer is not in the context, reply: *"I don't know based on provided data"*
4. Keep the answer concise and factual
5. Include names, dates, numbers exactly as they appear
6. End with source references: `Sources: [Source 1], [Source 2]`

---

### `generate_text(prompt, max_new_tokens)`

Runs the Qwen 2.5 language model to generate a response for the given prompt. Uses greedy decoding (no sampling) for deterministic, factual output.

```python
def generate_text(prompt: str, max_new_tokens: int = 256) -> str
```

**Parameters:**
- `prompt` — The full RAG prompt string
- `max_new_tokens` — Maximum tokens to generate (default: `256`)

**Returns:** The generated answer as a plain string

**How it works:**
1. Wraps the prompt in Qwen's chat template using `apply_chat_template()`
2. Tokenizes and moves inputs to the GPU
3. Calls `model.generate()` with `do_sample=False` and `temperature=0.0` for deterministic output
4. Decodes only the newly generated tokens (strips the input from output)

---

### `rag_answer(question)`

The main end-to-end RAG pipeline function. Takes a plain text question and returns a grounded answer with source citations.

```python
def rag_answer(question: str) -> Dict[str, Any]
```

**Parameters:**
- `question` — The user's question as a string

**Returns:** A dictionary with:
```python
{
    "question": str,   # The original question
    "answer": str,     # The generated answer
    "sources": [       # List of source chunks used
        {
            "source": str,    # Filename
            "chunk_id": int,  # Chunk index
            "score": float    # Similarity score
        }
    ]
}
```

**Pipeline steps:**
1. Calls `retrieve_chunks()` to get top-K matches
2. If no matches found → returns `"I don't know based on provided data"`
3. If top score is below `SIMILARITY_THRESHOLD (0.40)` → returns `"I don't know based on provided data"`
4. Calls `format_context()` to build context block
5. Calls `build_rag_prompt()` to build the full prompt
6. Calls `generate_text()` to generate the answer
7. Post-processes: if the model says it doesn't know, normalizes the response
8. Returns question, answer, and source metadata

---

### `transcribe_audio(audio_path)`

Transcribes a voice recording to text using OpenAI Whisper.

```python
def transcribe_audio(audio_path: str) -> str
```

**Parameters:**
- `audio_path` — Path to the audio file (`.m4a`, `.mp3`, `.wav`, etc.)

**Returns:** Transcribed text as a string

> Uses the Whisper `"base"` model by default. You can change `WHISPER_MODEL_NAME` to `"small"`, `"medium"`, or `"large"` in the config for better accuracy at the cost of speed.

---

### `ask_from_audio(audio_path)`

A convenience wrapper that combines `transcribe_audio()` and `rag_answer()` into a single call. Transcribes a voice question and returns the full RAG answer.

```python
def ask_from_audio(audio_path: str) -> Dict[str, Any]
```

**Parameters:**
- `audio_path` — Path to the audio file

**Returns:** Same dictionary as `rag_answer()`, plus an extra field:
```python
{
    "transcribed_question": str,  # What Whisper heard
    "question": str,
    "answer": str,
    "sources": [...]
}
```

---

## Example Usage

### Text Query

```python
result = rag_answer("Who is Ahmadullah?")
print(result["answer"])
# Ahmadullah is the chairman of As-Sunnah Foundation. He founded the
# organization in 2017 and is dedicated to education, da'wah, and
# overall humanitarian welfare.
```

### Voice Query

```python
result = ask_from_audio("recording.m4a")
print("Heard:", result["transcribed_question"])
print("Answer:", result["answer"])
```

---

## Models Used

| Model | Purpose | Size |
|---|---|---|
| `BAAI/bge-large-en-v1.5` | Text embedding for retrieval | ~1.3 GB |
| `Qwen/Qwen2.5-1.5B-Instruct` | Answer generation | ~3.1 GB |
| `openai/whisper-base` | Voice-to-text transcription | ~139 MB |

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (Tesla T4 or better)
- Pinecone account with a serverless index

---

## Notes

- The chatbot will only answer from the ingested documents. If the answer is not in your data, it will say so rather than guess.
- Run `C_Rag_Embed.ipynb` every time you add new documents to keep the Pinecone index up to date.
- The `google.colab` file upload cells are Colab-specific. For local use, replace them with direct file paths.
