import json
import re
from pathlib import Path


COURSE_CODE_PATTERN = re.compile(
    r"\bCS[\s-]*(\d{3})\b",
    flags=re.IGNORECASE
)

PREREQUISITE_PATTERN = re.compile(
    r"(?im)^[ \t]*Prerequisites?\s*:\s*([^\n]+)"
)


class CourseLookup:
    def __init__(self, jsonl_path):
        self.jsonl_path = Path(jsonl_path)

        if not self.jsonl_path.is_file():
            raise FileNotFoundError(
                f"Course documents not found: {self.jsonl_path}"
            )

        self.courses = {}

        with self.jsonl_path.open(
            "r", encoding="utf-8"
        ) as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                metadata = record["metadata"]
                code = metadata["course_code"]

                if code in self.courses:
                    raise ValueError(
                        f"Duplicate course entry: {code}"
                    )

                self.courses[code] = record

    @staticmethod
    def extract_course_codes(question):
        matches = COURSE_CODE_PATTERN.findall(question)

        return list(dict.fromkeys(
            f"CS{number}" for number in matches
        ))

    @staticmethod
    def detect_intent(question):
        q = question.lower()

        if re.search(
            r"(which|what)\s+courses?.*"
            r"(require|need|depend on)",
            q
        ):
            return "inverse_prerequisite"

        if "prerequisit" in q or "required before" in q:
            return "prerequisite"

        return "course_overview"

    @staticmethod
    def extract_prerequisite(record):
        match = PREREQUISITE_PATTERN.search(
            record["page_content"]
        )

        return match.group(1).strip() if match else None

    def get_course(self, code):
        return self.courses.get(code.upper())

    def get_prerequisites(self, code):
        record = self.get_course(code)

        if record is None:
            return None

        metadata = record["metadata"]
        prerequisite = self.extract_prerequisite(record)

        # An absent field is not equivalent to "None".
        if prerequisite is None:
            return None

        return {
            "course_code": metadata["course_code"],
            "course_title": metadata["course_title"],
            "prerequisites": prerequisite,
            "page": metadata["page"],
            "end_page": metadata["end_page"],
            "source": metadata["source"]
        }

    def get_description(self, code):
        record = self.get_course(code)

        if record is None:
            return None

        return {
            "content": record["page_content"],
            "metadata": record["metadata"]
        }

    def find_courses_requiring(self, target_code):
        target_code = re.sub(
            r"\s+", "", target_code
        ).upper()

        matches = []

        for record in self.courses.values():
            prerequisite = self.extract_prerequisite(record)

            if prerequisite is None:
                continue

            normalized = re.sub(
                r"\b([A-Za-z]+)[\s-]+(\d{3})\b",
                r"\1\2",
                prerequisite.upper()
            )

            if not re.search(
                rf"(?<![A-Z0-9])"
                rf"{re.escape(target_code)}"
                rf"(?![A-Z0-9])",
                normalized
            ):
                continue

            metadata = record["metadata"]

            matches.append({
                "course_code": metadata["course_code"],
                "course_title": metadata["course_title"],
                "prerequisites": prerequisite,
                "page": metadata["page"],
                "end_page": metadata["end_page"],
                "source": metadata["source"]
            })

        return sorted(
            matches,
            key=lambda item: item["course_code"]
        )
