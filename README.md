# Fulbright CS Handbook Assistant

**CS311 — Advanced Deep Learning | Final Project**  
**Project:** Building and Improving a Local Retrieval-Augmented Generation (RAG) System

A locally deployable academic handbook assistant that answers questions about the Fulbright University Vietnam **B.S. Computer Science Major Handbook (Version 1.0, May 2025)**. The project extends a basic text-based RAG workflow with PDF processing, structure-aware retrieval, deterministic course-information lookup, a Streamlit chat interface, and a Docker Compose deployment using Ollama.

> **Academic-use notice:** The assistant is a course project, not an official university advising service. Its handbook snapshot is dated May 2025. Confirm consequential academic decisions with current official university guidance.

## Features

- **PDF-backed RAG:** Uses a preprocessed Computer Science handbook and a persistent ChromaDB vector database.
- **Structure-aware retrieval:** Preserves selected section headings while indexing handbook chunks.
- **Course lookup:** Extracts course codes, titles, prerequisites, and descriptions from indexed course catalogue entries.
- **Inverse prerequisite search:** Finds indexed courses that explicitly list a named course as a prerequisite.
- **Question routing:** Uses deterministic lookup for supported course-specific questions and semantic retrieval plus Qwen for general handbook questions.
- **Conservative abstention:** Declines general questions when the nearest retrieved chunk exceeds an experimental distance threshold.
- **Source inspection:** Displays retrieved PDF page metadata in the Streamlit UI. These are retrieval references, **not independently verified claim-level citations**.
- **Local models:** The embedding model and language model run through Ollama; the app does not require an external language-model API.

## Architecture

```text
User (browser)
      |
      v
Streamlit UI (app/ui.py, localhost:8501)
      |
      v
AcademicChatbot (app/chatbot.py)
      |
      +-- Single recognized course code ------------------+
      |                                                    |
      |                                            CourseLookup
      |                                        (app/course_lookup.py)
      |                                                    |
      |                                  course_documents.jsonl
      |                                  - direct prerequisites
      |                                  - course descriptions
      |                                  - inverse prerequisites
      |
      +-- General handbook question / fallback -----------+
                                                           |
                                               HandbookRetriever
                                               (app/retrieval.py)
                                                           |
                                      Ollama nomic-embed-text --> ChromaDB
                                                           |
                                               Retrieved PDF chunks
                                                           |
                                               Ollama qwen2.5:0.5b
                                                           |
                                                     Answer + sources
```

**Docker Compose services:**

| Service | Responsibility |
| --- | --- |
| `chatbot` | Streamlit application, Python retrieval engine, local ChromaDB files |
| `ollama` | Embedding and language model inference, with persistent model volume |

ChromaDB is opened by the chatbot process from the bundled persistent directory; it is **not** a separate Compose service.

## Technology stack

- Python 3.11 (Docker image)
- Streamlit
- LangChain integrations (`langchain-chroma`, `langchain-ollama`)
- ChromaDB
- Ollama
- `nomic-embed-text` (embeddings)
- `qwen2.5:0.5b` (answer generation)
- Docker Desktop / Docker Engine with Docker Compose

## Repository structure

```text
CS311_Final_Project/
├── app/
│   ├── __init__.py
│   ├── chatbot.py           # Routing, conservative abstention, Qwen generation
│   ├── course_lookup.py     # Course and prerequisite extraction
│   ├── retrieval.py         # Persistent ChromaDB access and retrieval
│   └── ui.py                # Streamlit interface
├── data/
│   └── processed/
│       ├── course_documents.jsonl
│       └── structured_chunks.jsonl
├── vector_db/
│   └── structured/          # Prebuilt ChromaDB collection (62 vectors)
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
└── README.md
```

The course-aware experiment produced **20 detected course entries and 34 course-aware chunks**. The running application uses the exported `course_documents.jsonl` for direct lookup and the **62-vector structure-aware collection** for general RAG. The experimental course-specific vector collection is not required by the Docker application.

## Run locally with Docker (Windows, macOS, or Linux)

### Prerequisites

1. Install Docker Desktop (Windows/macOS) or Docker Engine and Compose (Linux).
2. Ensure the Docker daemon is running. Windows users should use the supported WSL 2 backend for Linux containers.
3. Clone this repository, or download and extract the repository folder.
4. Open PowerShell or a terminal **inside the folder containing `docker-compose.yml`**.

### 1. Build and start services

```bash
docker compose up -d --build
```

Check that both services are running:

```bash
docker compose ps
```

### 2. Download the required Ollama models

Run these commands once after the initial installation:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen2.5:0.5b
docker compose exec ollama ollama list
```

Both models must appear in the list. Docker stores them in the persistent `ollama_models` volume, so normal restarts do not require another download.

### 3. Open the application

Visit **http://localhost:8501**.

Try the following questions:

```text
What are the prerequisites for CS201?
Which courses require CS201?
How many credits are required for the Computer Science major?
What GPA is required to enroll in CS401 Capstone I?
```

Expected handbook facts include **CS101** as the prerequisite for CS201, **56 credits** for the CS major, and a **minimum major GPA of 3.5 plus approval of an application** for CS401 eligibility. Generated answers may omit relevant qualifications; consult the retrieved source text.

### 4. Stop and restart

Stop without deleting containers or model data:

```bash
docker compose stop
```

Restart existing containers:

```bash
docker compose start
```

Alternatively, start or recreate the stack if necessary:

```bash
docker compose up -d
```

Check status and model persistence:

```bash
docker compose ps
docker compose exec ollama ollama list
```

**Do not run `docker compose down -v` unless you intentionally want to delete the Ollama model volume.** Rebuilding with `docker compose up -d --build` is needed after changing application source files copied into the image.

### Troubleshooting

- **Cannot reach `localhost:8501`:** Run `docker compose ps` and inspect `docker compose logs chatbot`.
- **Ollama model missing:** Run the `ollama pull` commands above and check `docker compose exec ollama ollama list`.
- **Ollama connection error:** Check `docker compose logs ollama`. The app uses `http://ollama:11434` **inside Docker**, not `localhost:11434`.
- **Collection not found:** Check that `vector_db/structured/chroma.sqlite3` and its associated files exist in the repository. The collection is named `cs_handbook_structured`.
- **First general answer is slow:** Model loading and CPU inference can take longer than deterministic course lookup.
- **Blank UI or app exception:** Inspect `docker compose logs chatbot --tail=100` and confirm both services are running.

The Compose configuration binds the UI to `127.0.0.1:8501` on the host, so it is intended for access from the same computer rather than public deployment.

## Retrieval and generation approach

### Baseline

The starting workflow used a text-document RAG pipeline with character-based chunking, ChromaDB semantic retrieval, Ollama embeddings, and Qwen answer generation.

### PDF processing and structure-aware indexing

The handbook was parsed into page-level documents, chunked, and embedded. Structure-aware preprocessing introduced section labels such as **Major Requirements (56 credits)** and **Minor Requirements (24 credits)** to help retrieval locate academic requirements.

### Course-aware extraction and routing

The course catalogue was parsed into course entries using course heading patterns (for example, `CS201: Data Structures & Algorithms`). Course-specific metadata retained the course code, title, and PDF page range. For recognized single-course questions, a deterministic lookup extracts prerequisite fields or returns course descriptions. Inverse queries scan explicit prerequisite fields across indexed course entries.

This reduces a known semantic-search failure: for the question “What prerequisites are listed for CS201?”, a semantic-only query often ranked courses that **require** CS201 rather than the prerequisite **of** CS201.

### General handbook questions

For questions not handled by course lookup, `HandbookRetriever` embeds the question using `nomic-embed-text`, queries ChromaDB, and supplies top-ranked handbook chunks to Qwen. The current general-RAG branch applies an **experimental distance threshold of 0.60** and may abstain even when relevant evidence exists. Language-model outputs are marked as unverified in the UI.

## Selected experimental results

**Retrieval evaluation:** On 18 annotated answerable questions, the page-level benchmark gave the following results:

| Metric | Original semantic | Structure-aware |
| --- | ---: | ---: |
| Hit@1 | 0.7778 | 0.8333 |
| Hit@3 | 0.8333 | 0.9444 |
| Hit@5 | 1.0000 | 1.0000 |
| MRR@5 | 0.8389 | 0.8907 |
| All annotated evidence pages in Top-5 | 0.7222 | 0.7222 |

**Important interpretation:** These are **page-level** retrieval measures, not verified answer accuracy. Manual chunk-level review found cases where a hit came from a relevant page but lacked sufficient content to justify the answer.

**Diagnostic course retrieval:** A field-aware lookup correctly extracted direct prerequisites for CS201, CS203, and CS311 (3/3 selected tests). An inverse scan identified eight indexed course entries explicitly listing CS201 as a prerequisite. These small, hand-selected tests do not establish general accuracy.

**Abstention experiments:** A binary threshold of 0.68 correctly classified 9/10 selected stress questions, but incorrectly rejected an answerable question about capstone alternatives. A later conservative deployment setting of 0.60 intentionally prioritizes reduced unsupported generation at the cost of answer coverage. This setting has **not** been independently calibrated.

**Verification failure:** A Qwen-based evidence-verification prompt incorrectly conflated the major declaration requirement with capstone graduation alternatives, even though correct evidence was retrieved. Consequently, this experimental verifier was excluded from the deployed app.

**Illustrative observed response times:** A single general-RAG test took about 43.55 seconds on Google Colab CPU and 8.07 seconds on the developer's local Docker environment. These are **not controlled hardware benchmarks** and should not be interpreted as guaranteed speedups.

## Limitations

1. **Small language model:** Qwen 2.5 0.5B can omit details, repeat text, or confuse nearby requirements.
2. **False abstention:** A strict retrieval-distance gate can reject answerable paraphrases. The current 0.60 threshold is a heuristic, not proof of insufficient evidence.
3. **Citations:** The UI reports retrieved source pages, not verified claim-to-source entailment.
4. **Course coverage:** Rule-based course lookup covers 20 detected course entries; differently formatted or unindexed entries may be missed.
5. **Question routing:** Complex multi-course comparisons and unrecognized phrasing are not fully handled.
6. **Document version:** Academic requirements may have changed since May 2025.
7. **Chat history:** Streamlit session history is not persisted across application restarts.
8. **Local resources:** General-RAG performance depends on available CPU/RAM and whether Ollama models are loaded.
9. **Validation scope:** The observed tests are not a comprehensive evaluation of deployed answer correctness.

## Demonstrations and report

### Demo Videos

The following videos demonstrate the main components of the CS311 Final Project.

- **PDF Loading and Retrieval Demo:** [Watch Video 1](https://drive.google.com/file/d/1hfOx72fcMOZhWRPtuhWjv2z_KUII4Qcp/view?usp=sharing)
- **Major Handbook Chatbot Demo:** [Watch Video 2](https://drive.google.com/file/d/1rJOM6Z2xf7jD2tJ6_1wHepsZDSk11u86/view?usp=sharing)
- **Docker Deployment Demo:** [Watch Video 3](https://drive.google.com/file/d/1HCgBb1ridXjUeWuZ1aeGooNIUMaFovPS/view?usp=sharing)
- **Streamlit UI/UX Demo:** [Watch Video 4](https://drive.google.com/file/d/15do7anKur9kUh3rufgeSHoD3eR3hfYz-/view?usp=sharing)

- Final technical report: **[link to be added]**

## Reproducibility notes

The repository bundles precomputed handbook vectors so first-time Docker users do not have to re-embed the PDF. To reproduce the full preprocessing and experimental pipeline, consult the accompanying research notebook if included in the final submission. Ensure any redistributed handbook content is authorized for sharing.

## License and attribution

This is an academic coursework project. The handbook and course descriptions belong to their respective rights holders. No license for redistribution of the university handbook is implied by this README.
