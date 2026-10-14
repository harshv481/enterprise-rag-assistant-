"""Source-grounded response generation without external paid LLMs."""
from typing import List, Tuple
from src.models import RetrievedChunk, Citation


class GroundedGenerator:
    """
    Template-based grounded generator that synthesizes enterprise answers
    directly from retrieved source passages with strict source attribution.
    """

    def generate(
        self, query: str, user_role: str, retrieved_chunks: List[RetrievedChunk]
    ) -> Tuple[str, List[Citation]]:
        citations: List[Citation] = []

        if not retrieved_chunks:
            answer = (
                f"### Access Denied or No Information Found\n\n"
                f"No accessible enterprise records were found for query: *\"{query}\"* under role **'{user_role}'**.\n"
                f"- Ensure your user role has authorized clearance for this document category.\n"
                f"- Check that your search terms match internal company terminology."
            )
            return answer, []

        for item in retrieved_chunks:
            chunk = item.chunk
            citations.append(
                Citation(
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    section_title=chunk.section_title,
                    chunk_id=chunk.chunk_id,
                    relevance_score=item.score,
                )
            )

        top_chunk = retrieved_chunks[0].chunk
        doc_names = list(dict.fromkeys([c.chunk.doc_title for c in retrieved_chunks]))

        answer_lines = [
            f"### Verified Enterprise Summary",
            f"Based on internal documentation (**{', '.join(doc_names)}**), authorized for role `{user_role}`:",
            "",
            f"**Key Finding ({top_chunk.section_title}):**",
            f"> {top_chunk.content.splitlines()[-1] if len(top_chunk.content.splitlines()) > 1 else top_chunk.content}",
            "",
            "#### Detailed Provisions:",
        ]

        for i, item in enumerate(retrieved_chunks, start=1):
            chunk = item.chunk
            # Format clean snippet
            snippet = chunk.content.replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:277] + "..."
            answer_lines.append(
                f"- **[{chunk.doc_title} - {chunk.section_title}]** (Relevance: {item.score:.2f}):\n"
                f"  \"{snippet}\""
            )

        answer_lines.append("")
        answer_lines.append(
            f"*Source-grounded response synthesized across {len(retrieved_chunks)} internal chunks. "
            f"Clearance level: `{user_role}`.*"
        )

        return "\n".join(answer_lines), citations
