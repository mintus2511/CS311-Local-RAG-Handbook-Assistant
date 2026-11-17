import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class HandbookRetriever:
    def __init__(
        self,
        db_path,
        collection_name="cs_handbook_structured",
        embedding_model="nomic-embed-text",
        ollama_base_url=None
    ):
        self.db_path = Path(db_path)
        self.collection_name = collection_name

        self.ollama_base_url = (
            ollama_base_url
            or os.getenv(
                "OLLAMA_BASE_URL",
                "http://127.0.0.1:11434"
            )
        )

        if not self.db_path.is_dir():
            raise FileNotFoundError(
                f"ChromaDB directory not found: {self.db_path}"
            )

        if not (self.db_path / "chroma.sqlite3").is_file():
            raise FileNotFoundError(
                "Existing ChromaDB data not found at "
                f"{self.db_path}. Refusing to create "
                "an empty index."
            )

        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=self.ollama_base_url
        )

        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.db_path),
            create_collection_if_not_exists=False
        )

        self.document_count = self.db._collection.count()

        if self.document_count == 0:
            raise RuntimeError(
                "The specified Chroma collection is empty."
            )

    def search(self, question, top_k=5):
        results = self.db.similarity_search_with_score(
            question,
            k=top_k
        )

        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "section": doc.metadata.get("section"),
                "distance": float(distance)
            }
            for doc, distance in results
        ]

    @staticmethod
    def build_context(results):
        sections = []

        for rank, result in enumerate(results, start=1):
            sections.append(
                f"[Source {rank}: "
                f"{result['source']}, "
                f"PDF page {result['page']}]\n"
                f"{result['content']}"
            )

        return "\n\n---\n\n".join(sections)
