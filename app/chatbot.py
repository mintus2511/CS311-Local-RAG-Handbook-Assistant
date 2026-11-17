import os
import time
from pathlib import Path

from langchain_ollama import OllamaLLM

from app.course_lookup import CourseLookup
from app.retrieval import HandbookRetriever


ABSTENTION_MESSAGE = (
    "I could not find sufficiently reliable evidence "
    "to answer this question from the provided "
    "Computer Science Major Handbook."
)


class AcademicChatbot:
    def __init__(
        self,
        project_root,
        ollama_base_url=None,
        llm_model="qwen2.5:0.5b",
        embedding_model="nomic-embed-text",
        distance_threshold=0.60
    ):
        self.project_root = Path(project_root)

        self.ollama_base_url = (
            ollama_base_url
            or os.getenv(
                "OLLAMA_BASE_URL",
                "http://127.0.0.1:11434"
            )
        )

        self.distance_threshold = distance_threshold

        self.courses = CourseLookup(
            self.project_root
            / "data/processed/course_documents.jsonl"
        )

        self.retriever = HandbookRetriever(
            db_path=(
                self.project_root
                / "vector_db/structured"
            ),
            collection_name="cs_handbook_structured",
            embedding_model=embedding_model,
            ollama_base_url=self.ollama_base_url
        )

        self.llm = OllamaLLM(
            model=llm_model,
            base_url=self.ollama_base_url,
            temperature=0,
            num_predict=220
        )

    def _make_result(
        self, answer, route, grounding,
        sources, start, **extra
    ):
        return {
            "answer": answer,
            "route": route,
            "grounding": grounding,
            "sources": sources,
            "elapsed": time.perf_counter() - start,
            **extra
        }

    def ask(self, question, top_k=5):
        start = time.perf_counter()

        question = question.strip()

        if not question:
            return self._make_result(
                "Please enter a question.",
                "invalid_input",
                "not_applicable",
                [],
                start
            )

        course_codes = self.courses.extract_course_codes(
            question
        )
        intent = self.courses.detect_intent(question)

        # Route A: Course-specific questions
        if len(course_codes) == 1:
            code = course_codes[0]
            course_record = self.courses.get_course(code)

            # Only use deterministic lookup when the
            # course exists in the extracted catalogue.
            if course_record is not None:

                if intent == "inverse_prerequisite":
                    matches = (
                        self.courses.find_courses_requiring(
                            code
                        )
                    )

                    if matches:
                        lines = [
                            (
                                f"{item['course_code']}: "
                                f"{item['course_title']} — "
                                f"Prerequisites: "
                                f"{item['prerequisites']} "
                                f"[PDF pp. {item['page']}-"
                                f"{item['end_page']}]"
                            )
                            for item in matches
                        ]

                        answer = (
                            f"Courses explicitly listing "
                            f"{code} as a prerequisite "
                            "in the indexed catalogue:\n"
                            + "\n".join(lines)
                        )
                    else:
                        answer = (
                            f"No direct dependencies on {code} "
                            "were found among the indexed "
                            "course entries."
                        )

                    return self._make_result(
                        answer,
                        "inverse_prerequisite",
                        "deterministic_extraction",
                        matches,
                        start
                    )

                if intent == "prerequisite":
                    item = self.courses.get_prerequisites(
                        code
                    )

                    if item is not None:
                        answer = (
                            f"{item['course_code']}: "
                            f"{item['course_title']}\n"
                            f"Prerequisites: "
                            f"{item['prerequisites']}"
                        )

                        return self._make_result(
                            answer,
                            "course_lookup",
                            "deterministic_extraction",
                            [item],
                            start
                        )

                elif intent == "course_overview":
                    item = self.courses.get_description(
                        code
                    )

                    if item is not None:
                        return self._make_result(
                            item["content"],
                            "course_lookup",
                            "deterministic_extraction",
                            [item["metadata"]],
                            start
                        )

            # If no suitable structured record exists,
            # search other sections of the handbook.
            # For inverse questions, do not substitute
            # semantic similarity for exhaustive lookup.
            if intent == "inverse_prerequisite":
                return self._make_result(
                    ABSTENTION_MESSAGE,
                    "inverse_prerequisite",
                    "abstained",
                    [],
                    start
                )

        # Route B: General semantic RAG
        results = self.retriever.search(
            question, top_k=top_k
        )

        if not results:
            return self._make_result(
                ABSTENTION_MESSAGE,
                "general_rag",
                "abstained",
                [],
                start
            )

        top1_distance = results[0]["distance"]

        if top1_distance > self.distance_threshold:
            return self._make_result(
                ABSTENTION_MESSAGE,
                "general_rag",
                "abstained",
                [],
                start,
                top1_distance=top1_distance
            )

        context = self.retriever.build_context(results)

        prompt = f"""You are an academic handbook assistant.

Answer ONLY using the supplied handbook excerpts.
Do not invent missing facts.

If the excerpts do not contain sufficient information,
say: "The provided handbook does not specify this."

Answer briefly and accurately.
Do not confuse major declaration requirements with
degree completion requirements.

Handbook excerpts:
{context}

Student question:
{question}

Answer:
"""

        answer = self.llm.invoke(prompt).strip()

        sources = [
            {
                "source": item["source"],
                "page": item["page"],
                "section": item["section"],
                "distance": item["distance"]
            }
            for item in results
        ]

        return self._make_result(
            answer,
            "general_rag",
            "llm_generated_unverified",
            sources,
            start,
            top1_distance=top1_distance
        )
