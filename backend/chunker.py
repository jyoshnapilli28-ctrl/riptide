"""
RIPTIDE Dynamic Structure-Aware Email Chunker.
==============================================
STRICT REQUIREMENT: FIXED-SIZE CHUNKING IS FORBIDDEN.

This module implements structure-aware dynamic chunking over email communications.
It inspects:
  1. RFC 822 / MIME Header Envelope (From, To, Date, Subject, Folder, etc.)
  2. Thread boundaries (Original Message, Forwarded by, Quoted threads)
  3. Signature blocks and confidentiality disclaimers
  4. Natural paragraph boundaries
  5. Sentence-level semantic clusters for long thematic paragraphs
  6. Structured lists and itemized content

Every chunk outputs rich audit metadata:
  - email_id
  - chunk_id (e.g. {email_id}_c001)
  - chunk_type (header_summary, body_paragraph, forwarded_thread, quoted_reply, signature_block, list_block)
  - chunk_reason (rfc822_header_envelope, natural_paragraph_boundary, thread_forward_boundary, etc.)
  - source
  - text
  - char_count
  - token_estimate
"""

import re
from typing import List, Dict, Any, Optional, Tuple


# Regex patterns for structural boundaries
HEADER_SPLIT_REGEX = re.compile(r"\r?\n\r?\n")
THREAD_FORWARD_REGEX = re.compile(
    r"(?:-{3,}\s*(?:Original Message|Forwarded by|Forwarded Message)\s*-{3,}"
    r"|From:\s*.*?\r?\n(?:Sent|Date):\s*.*?\r?\n(?:To|Subject):"
    r"|---------------------- Forwarded by)",
    re.IGNORECASE,
)
SIGNATURE_REGEX = re.compile(
    r"(?:\r?\n--\s*\r?\n"
    r"|\r?\n(?:Best regards|Kind regards|Regards|Thanks|Thank you|Sincerely|Cheers),\s*\r?\n"
    r"|\r?\n(?:CONFIDENTIALITY NOTICE|This e-mail and any attachments|This message is intended for the addressee))",
    re.IGNORECASE,
)
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
LIST_ITEM_REGEX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)


class DynamicEmailChunker:
    """Structure-aware dynamic chunker for email archives."""

    def __init__(
        self,
        min_chunk_chars: int = 40,
        target_chunk_chars: int = 500,
        max_chunk_chars: int = 1200,
    ):
        self.min_chunk_chars = min_chunk_chars
        self.target_chunk_chars = target_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    def chunk(
        self,
        raw_email: str,
        email_id: str,
        source: str = "enron_corpus",
    ) -> List[Dict[str, Any]]:
        """
        Dynamically chunks an email using structural boundary analysis.
        No fixed-size windowing is used.
        """
        if not raw_email or not raw_email.strip():
            return []

        chunks: List[Dict[str, Any]] = []
        chunk_counter = 0

        # Helper to append formatted chunk
        def add_chunk(text: str, chunk_type: str, chunk_reason: str):
            nonlocal chunk_counter
            cleaned = text.strip()
            if not cleaned or len(cleaned) < 15:
                return
            chunk_counter += 1
            chunk_id = f"{email_id}_c{chunk_counter:03d}"
            chunks.append(
                {
                    "email_id": email_id,
                    "chunk_id": chunk_id,
                    "chunk_type": chunk_type,
                    "chunk_reason": chunk_reason,
                    "source": source,
                    "char_count": len(cleaned),
                    "token_estimate": len(cleaned.split()),
                    "text": cleaned,
                }
            )

        # 1. Separate RFC 822 Header Block from Body
        header_text, body_text = self._extract_header_envelope(raw_email)

        if header_text:
            parsed_headers = self._parse_rfc822_headers(header_text)
            if parsed_headers:
                header_summary = self._format_header_summary(parsed_headers)
                add_chunk(
                    header_summary,
                    chunk_type="header_summary",
                    chunk_reason="rfc822_header_envelope",
                )

        if not body_text or not body_text.strip():
            # If email had only headers or body extraction yielded nothing
            if not chunks and header_text:
                add_chunk(
                    header_text,
                    chunk_type="header_summary",
                    chunk_reason="rfc822_header_raw",
                )
            return chunks

        # 2. Extract Signatures / Disclaimers at the bottom of the body
        body_main, sig_text, sig_reason = self._extract_signature_block(body_text)

        # 3. Detect Thread Transitions (Forwarded / Quoted chains)
        thread_segments = self._split_threads(body_main)

        # 4. Process each thread segment into dynamic structural chunks
        for segment_idx, (seg_text, seg_kind, seg_reason) in enumerate(thread_segments):
            if seg_kind in ("forwarded_thread", "quoted_reply"):
                # If forwarded thread contains its own mini-envelope, split it
                sub_header, sub_body = self._extract_header_envelope(seg_text)
                if sub_header:
                    add_chunk(
                        sub_header,
                        chunk_type="forwarded_thread",
                        chunk_reason=f"{seg_reason}:embedded_envelope",
                    )
                    self._chunk_body_paragraphs(
                        sub_body,
                        add_chunk,
                        base_type=seg_kind,
                        base_reason=seg_reason,
                    )
                else:
                    self._chunk_body_paragraphs(
                        seg_text,
                        add_chunk,
                        base_type=seg_kind,
                        base_reason=seg_reason,
                    )
            else:
                self._chunk_body_paragraphs(
                    seg_text,
                    add_chunk,
                    base_type="body_paragraph",
                    base_reason=seg_reason or "natural_body_flow",
                )

        # 5. Append Signature Chunk if present and meaningful
        if sig_text and len(sig_text.strip()) >= 20:
            add_chunk(
                sig_text,
                chunk_type="signature_block",
                chunk_reason=sig_reason or "boundary:signature_delimiter",
            )

        # Edge case: If no chunks produced yet (e.g. extremely short body), emit whole body
        if not chunks and body_text.strip():
            add_chunk(
                body_text,
                chunk_type="body_paragraph",
                chunk_reason="single_short_body_fallback",
            )

        return chunks

    def _extract_header_envelope(self, text: str) -> Tuple[str, str]:
        """Separates top RFC 822 headers from the body."""
        # Look for first blank line separating headers from body
        match = HEADER_SPLIT_REGEX.search(text)
        if match:
            potential_header = text[: match.start()].strip()
            # Verify if it looks like RFC 822 headers (contains Message-ID:, From:, Date:, or Subject:)
            header_keys = ["message-id:", "from:", "to:", "subject:", "date:", "x-from:", "x-to:"]
            lower_head = potential_header.lower()
            if any(k in lower_head for k in header_keys):
                body = text[match.end() :]
                return potential_header, body

        # If no RFC header matched
        return "", text

    def _parse_rfc822_headers(self, header_block: str) -> Dict[str, str]:
        """Parses raw header lines into key-value map."""
        headers: Dict[str, str] = {}
        current_key: Optional[str] = None
        current_val: List[str] = []

        for line in header_block.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if (line.startswith(" ") or line.startswith("\t")) and current_key:
                current_val.append(line_stripped)
            elif ":" in line:
                if current_key:
                    headers[current_key] = " ".join(current_val).strip()
                colon_idx = line.find(":")
                current_key = line[:colon_idx].strip()
                current_val = [line[colon_idx + 1 :].strip()]
            else:
                if current_key:
                    current_val.append(line_stripped)

        if current_key:
            headers[current_key] = " ".join(current_val).strip()

        return headers

    def _format_header_summary(self, headers: Dict[str, str]) -> str:
        """Formats important headers for semantic retrieval."""
        keys_to_include = [
            ("Subject", headers.get("Subject") or headers.get("subject")),
            ("From", headers.get("From") or headers.get("from") or headers.get("X-From")),
            ("To", headers.get("To") or headers.get("to") or headers.get("X-To")),
            ("Date", headers.get("Date") or headers.get("date")),
            ("Cc", headers.get("Cc") or headers.get("cc") or headers.get("X-cc")),
            ("Folder", headers.get("X-Folder") or headers.get("Folder")),
        ]
        lines = []
        for label, val in keys_to_include:
            if val and val.strip():
                lines.append(f"{label}: {val.strip()}")
        return "\n".join(lines) if lines else "Email Header: [Unspecified]"

    def _extract_signature_block(self, body: str) -> Tuple[str, str, str]:
        """Detects standard signature blocks or legal disclaimers at end of body."""
        matches = list(SIGNATURE_REGEX.finditer(body))
        if not matches:
            return body, "", ""

        # Take the last prominent signature trigger
        last_match = matches[-1]
        sig_start = last_match.start()
        # Only treat as signature if it occurs in the lower 60% of the body
        if sig_start > len(body) * 0.4:
            main_body = body[:sig_start]
            sig_content = body[sig_start:]
            matched_str = last_match.group(0).strip().lower()
            if "confidential" in matched_str or "disclaimer" in matched_str:
                reason = "boundary:confidentiality_notice"
            else:
                reason = "boundary:signature_delimiter"
            return main_body, sig_content, reason

        return body, "", ""

    def _split_threads(self, body: str) -> List[Tuple[str, str, str]]:
        """Splits body into primary communication and forwarded/quoted thread segments."""
        matches = list(THREAD_FORWARD_REGEX.finditer(body))
        if not matches:
            return [(body, "body_paragraph", "initial_thread")]

        segments = []
        prev_idx = 0

        for match in matches:
            start, end = match.span()
            if start > prev_idx:
                prior_text = body[prev_idx:start].strip()
                if prior_text:
                    segments.append((prior_text, "body_paragraph", "thread_lead_segment"))
            # Thread separator matched
            prev_idx = start
            break  # Treat everything from first forward/original message as forwarded section

        if prev_idx < len(body):
            fwd_text = body[prev_idx:].strip()
            if fwd_text:
                segments.append((fwd_text, "forwarded_thread", "thread_boundary:forwarded_message"))

        return segments if segments else [(body, "body_paragraph", "initial_thread")]

    def _chunk_body_paragraphs(
        self,
        text: str,
        add_chunk_fn,
        base_type: str = "body_paragraph",
        base_reason: str = "natural_paragraph_boundary",
    ):
        """
        Dynamically groups paragraphs, sentences, and lists without fixed sizing.
        Uses structural break points (double newlines, list patterns, sentence clusters).
        """
        # Split on natural paragraph breaks (\n\s*\n)
        raw_paras = re.split(r"\r?\n\s*\r?\n", text)
        buffer_text = ""
        buffer_reason = ""

        for para in raw_paras:
            para_clean = para.strip()
            if not para_clean:
                continue

            # Check if this paragraph is a list / itemized block
            is_list = bool(LIST_ITEM_REGEX.search(para_clean))
            current_type = "list_block" if is_list else base_type

            # If paragraph is very long (> max_chunk_chars), split by sentence clusters
            if len(para_clean) > self.max_chunk_chars:
                # Flush existing buffer first
                if buffer_text:
                    add_chunk_fn(buffer_text, base_type, buffer_reason or base_reason)
                    buffer_text = ""
                    buffer_reason = ""

                self._chunk_sentences(
                    para_clean,
                    add_chunk_fn,
                    base_type=current_type,
                    base_reason="sentence_cluster_topic_continuity",
                )
                continue

            # If small paragraph (e.g. conversational single-line or short question/answer)
            if len(para_clean) < self.min_chunk_chars:
                if buffer_text:
                    buffer_text += "\n\n" + para_clean
                    buffer_reason = "cohesive_short_paragraphs"
                else:
                    buffer_text = para_clean
                    buffer_reason = "short_initial_paragraph"

                # If accumulated buffer reaches healthy size, emit
                if len(buffer_text) >= self.target_chunk_chars:
                    add_chunk_fn(buffer_text, current_type, buffer_reason)
                    buffer_text = ""
                    buffer_reason = ""
            else:
                # Flush prior buffer if substantial
                if buffer_text:
                    add_chunk_fn(buffer_text, base_type, buffer_reason or base_reason)
                    buffer_text = ""
                    buffer_reason = ""

                # Emit the cohesive paragraph as its own chunk
                add_chunk_fn(
                    para_clean,
                    chunk_type=current_type,
                    chunk_reason="natural_paragraph_boundary",
                )

        # Flush any remaining buffer
        if buffer_text:
            add_chunk_fn(buffer_text, base_type, buffer_reason or base_reason)

    def _chunk_sentences(
        self,
        text: str,
        add_chunk_fn,
        base_type: str,
        base_reason: str,
    ):
        """Splits long paragraphs into cohesive sentence groups (semantic continuity)."""
        sentences = SENTENCE_SPLIT_REGEX.split(text)
        if len(sentences) <= 1:
            add_chunk_fn(text, base_type, f"{base_reason}:unsplit_long_paragraph")
            return

        current_cluster = []
        current_len = 0

        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue

            current_cluster.append(sent_clean)
            current_len += len(sent_clean) + 1

            if current_len >= self.target_chunk_chars:
                chunk_str = " ".join(current_cluster)
                add_chunk_fn(chunk_str, base_type, base_reason)
                current_cluster = []
                current_len = 0

        if current_cluster:
            chunk_str = " ".join(current_cluster)
            add_chunk_fn(chunk_str, base_type, f"{base_reason}:final_sentence_cluster")


# Global singleton helper
_default_chunker = DynamicEmailChunker()


def chunk_email(
    raw_email: str,
    email_id: str,
    source: str = "enron_corpus",
) -> List[Dict[str, Any]]:
    """Convenience function for dynamic structure-aware email chunking."""
    return _default_chunker.chunk(raw_email=raw_email, email_id=email_id, source=source)
