"""Document Store and Ingestion Engine with Role-Based Access Control (RBAC)."""
import os
import re
from typing import List, Dict, Optional
from src.models import DocumentChunk

# Role definitions for enterprise documents
DOCUMENT_ROLE_MAP: Dict[str, List[str]] = {
    "hr_remote_work_policy.md": [
        "employee", "manager", "hr", "it_admin", "finance", "executive"
    ],
    "hr_compensation_and_benefits.md": [
        "hr", "executive"
    ],
    "it_vpn_and_security_guidelines.md": [
        "employee", "manager", "hr", "it_admin", "finance", "executive"
    ],
    "it_admin_infrastructure_access.md": [
        "it_admin"
    ],
    "finance_travel_expense_policy.md": [
        "employee", "manager", "finance", "executive"
    ],
    "executive_strategic_roadmap_2026.md": [
        "executive"
    ],
}


class DocumentStore:
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = docs_dir
        self.chunks: List[DocumentChunk] = []
        self.doc_registry: Dict[str, Dict] = {}
        self.load_documents()

    def load_documents(self) -> None:
        """Parse and chunk markdown documents in the target folder."""
        self.chunks.clear()
        self.doc_registry.clear()

        if not os.path.exists(self.docs_dir):
            return

        for filename in sorted(os.listdir(self.docs_dir)):
            if not filename.endswith((".md", ".txt")):
                continue

            file_path = os.path.join(self.docs_dir, filename)
            allowed_roles = DOCUMENT_ROLE_MAP.get(
                filename, ["employee", "manager", "executive"]
            )

            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            parsed_chunks = self._chunk_markdown(filename, raw_text, allowed_roles)
            self.chunks.extend(parsed_chunks)

            self.doc_registry[filename] = {
                "file_path": file_path,
                "total_chunks": len(parsed_chunks),
                "allowed_roles": allowed_roles,
            }

    def _chunk_markdown(
        self, filename: str, content: str, allowed_roles: List[str]
    ) -> List[DocumentChunk]:
        """Split markdown files by section headings."""
        chunks: List[DocumentChunk] = []

        # Extract main title
        lines = content.splitlines()
        main_title = filename
        for line in lines:
            if line.startswith("# "):
                main_title = line.replace("# ", "").strip()
                break

        # Split on markdown h2 headings: ## <Title>
        sections = re.split(r"(?m)^##\s+", content)

        if len(sections) <= 1:
            # Fallback to paragraph splitting if no h2 headings
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for idx, p in enumerate(paragraphs):
                chunk_id = f"{filename}#chunk_{idx + 1}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=filename,
                        doc_title=main_title,
                        section_title=f"Section {idx + 1}",
                        content=p,
                        allowed_roles=allowed_roles,
                    )
                )
            return chunks

        # Section 0 may contain introduction or h1
        intro = sections[0].strip()
        if intro and len(intro.split()) > 10:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{filename}#intro",
                    doc_id=filename,
                    doc_title=main_title,
                    section_title="Introduction",
                    content=intro,
                    allowed_roles=allowed_roles,
                )
            )

        # Subsequent sections have title on first line, rest is content
        for idx, sec in enumerate(sections[1:], start=1):
            sec_lines = sec.strip().splitlines()
            section_title = sec_lines[0].strip() if sec_lines else f"Section {idx}"
            section_body = "\n".join(sec_lines[1:]).strip() if len(sec_lines) > 1 else ""

            full_chunk_text = f"{section_title}\n{section_body}".strip()
            chunk_id = f"{filename}#sec_{idx}"

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=filename,
                    doc_title=main_title,
                    section_title=section_title,
                    content=full_chunk_text,
                    allowed_roles=allowed_roles,
                )
            )

        return chunks

    def get_chunks_for_role(self, user_role: str) -> List[DocumentChunk]:
        """Filter chunks based on RBAC permissions."""
        role_normalized = user_role.lower().strip()
        return [
            chunk for chunk in self.chunks
            if role_normalized in [r.lower() for r in chunk.allowed_roles]
        ]

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Return all document chunks."""
        return self.chunks

    def get_summary(self) -> Dict:
        """Return stats on ingested documents and roles."""
        return {
            "total_documents": len(self.doc_registry),
            "total_chunks": len(self.chunks),
            "documents": self.doc_registry,
        }
