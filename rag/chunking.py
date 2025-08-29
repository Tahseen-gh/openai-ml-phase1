from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(?P<title>.+)$", re.M)
_SPLIT_RE = re.compile(r"(?<=\S)(?:(?<=[.!?])\s+|\n{2,})")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    text: str
    start: int  # absolute char offset (inclusive)
    end: int  # absolute char offset (exclusive)
    heading: str | None = None


def _heading_sections(text: str) -> list[tuple[int, int, str | None]]:
    matches = list(_HEADING_RE.finditer(text))
    sections: list[tuple[int, int, str | None]] = []
    if not matches:
        return [(0, len(text), None)]
    first = matches[0]
    if first.start() > 0:
        sections.append((0, first.start(), None))
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = m.group("title").strip()
        sections.append((body_start, body_end, title))
    return sections


def _sent_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SPLIT_RE.finditer(text):
        end = m.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = m.end()
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _label_for_offset(sections: list[tuple[int, int, str | None]], offset: int) -> str | None:
    for s, e, title in sections:
        if s <= offset < e:
            return title
    return None


def _heading_for_chunk(
    sections: list[tuple[int, int, str | None]], start: int, end: int
) -> str | None:
    # 1) If start is already inside a section body, use that
    label = _label_for_offset(sections, start)
    if label:
        return label
    # 2) Otherwise, if the chunk spans into the next section body, use that section's title
    for s, _, title in sections:
        if start < s <= end:
            return title
    return None


def _keep_suffix(
    spans: list[tuple[int, int]], keep_chars: int
) -> tuple[list[tuple[int, int]], int]:
    if keep_chars <= 0 or not spans:
        return ([], 0)
    total = sum(e - s for s, e in spans)
    if keep_chars >= total:
        return (spans[:], total)
    out: list[tuple[int, int]] = []
    tally = 0
    for s, e in reversed(spans):
        seg = e - s
        if tally + seg >= keep_chars:
            take = keep_chars - tally
            if take > 0:
                out.insert(0, (e - take, e))
                tally += take
            break
        else:
            out.insert(0, (s, e))
            tally += seg
    return (out, tally)


def chunk_text(doc_id: str, text: str, *, max_chars: int = 800, overlap: int = 120) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    eff_overlap = min(overlap, max_chars)
    sections = _heading_sections(text)
    spans = _sent_spans(text)
    if not spans:
        return [Chunk(doc_id=doc_id, text=text, start=0, end=len(text), heading=None)]

    chunks: list[Chunk] = []
    buf: list[tuple[int, int]] = []
    buf_chars = 0

    def flush():
        nonlocal buf, buf_chars
        if not buf:
            return
        # Build body from the spans only (no separators) so len(body) == buf_chars
        body = "".join(text[s:e] for s, e in buf).strip()
        if not body:
            buf, buf_chars = [], 0
            return
        s0, e_n = buf[0][0], buf[-1][1]
        heading = _heading_for_chunk(sections, s0, e_n)
        chunks.append(Chunk(doc_id=doc_id, text=body, start=s0, end=e_n, heading=heading))
        buf, buf_chars = _keep_suffix(buf, eff_overlap)

    for s, e in spans:
        span_len = e - s

        # If a single span is too long, hard-split it into windows of size max_chars
        if span_len > max_chars:
            flush()
            i = s
            while i < e:
                j = min(i + max_chars, e)
                body = text[i:j].strip()
                if body:
                    heading = _heading_for_chunk(sections, i, j)
                    chunks.append(Chunk(doc_id=doc_id, text=body, start=i, end=j, heading=heading))
                if j >= e:
                    break
                i = j - eff_overlap if eff_overlap > 0 else j
            buf, buf_chars = [], 0
            continue

        if not buf:
            buf.append((s, e))
            buf_chars = span_len
            continue

        if buf_chars + span_len <= max_chars:
            buf.append((s, e))
            buf_chars += span_len
        else:
            flush()
            keep = max_chars - span_len
            if keep < 0:
                keep = 0
            if buf_chars > keep:
                buf, buf_chars = _keep_suffix(buf, keep)
            buf.append((s, e))
            buf_chars += span_len

    flush()
    return chunks
