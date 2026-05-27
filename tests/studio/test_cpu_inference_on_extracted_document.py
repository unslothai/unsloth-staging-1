# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""PR-5351 cross-OS CPU-inference smoke test.

End-to-end: extract a small public PDF locally (no network during
extraction), then feed the extracted markdown into a tiny GGUF via
llama-cpp-python on CPU and assert the model identifies the document.

Runs on ubuntu-latest / macos-14 / windows-latest GitHub-Actions
runners. CPU-only; no real GPU is required because the test path
imports `_extract_pdf` directly and runs llama-cpp-python's CPU build.
"""

from __future__ import annotations

import importlib
import io
import os
import sys
import textwrap
from pathlib import Path

import pytest


def _make_text_pdf(body: str) -> bytes:
    """Build a tiny one-page PDF whose stream is the literal `body`.

    Avoids pulling a real LaTeX/wkhtmltopdf chain into CI -- the PR's
    pymupdf-based extractor recovers the text via its standard pdfminer
    fallback path even without a content-stream filter.
    """
    pdf = io.BytesIO()
    pdf.write(b"%PDF-1.4\n")
    objects = []

    def write(obj_bytes: bytes) -> int:
        offset = pdf.tell()
        objects.append(offset)
        pdf.write(obj_bytes)
        return len(objects)

    write(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    write(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    write(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    text_stream = (
        "BT\n/F1 12 Tf\n72 720 Td\n"
        + "\n".join(
            f"({line}) Tj T* "
            for line in body.splitlines()
            if line.strip()
        )
        + "\nET\n"
    )
    stream_bytes = text_stream.encode("latin-1", errors="replace")
    write(
        f"4 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
        + stream_bytes
        + b"\nendstream\nendobj\n"
    )
    write(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    xref_offset = pdf.tell()
    pdf.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in objects:
        pdf.write(f"{off:010d} 00000 n \n".encode())
    pdf.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return pdf.getvalue()


@pytest.fixture(scope="module")
def extractor():
    """Import the PR's `_extract_pdf` directly so this is a unit-level
    test of the extractor + a CPU integration test of llama-cpp-python."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "studio" / "backend"))
    mod = importlib.import_module("core.chat.document_extractor")
    return mod._extract_pdf


@pytest.fixture(scope="module")
def llama():
    """Load a tiny GGUF on CPU. Skips if llama-cpp-python isn't installed."""
    pytest.importorskip("llama_cpp")
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    cache_dir = Path(os.environ.get("PR5351_GGUF_CACHE", str(Path.home() / ".cache" / "pr5351_gguf")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Tiny instruction-tuned model that fits 7 GB CPU runners.
    repo = "unsloth/Qwen2.5-0.5B-Instruct-GGUF"
    fname = "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
    path = hf_hub_download(
        repo_id=repo,
        filename=fname,
        local_dir=str(cache_dir),
    )
    return Llama(
        model_path=path,
        n_ctx=4096,
        n_threads=int(os.environ.get("PR5351_LLAMA_THREADS", "2")),
        verbose=False,
    )


@pytest.mark.timeout(900)
def test_cpu_inference_identifies_extracted_document(extractor, llama, tmp_path):
    """Extract a synthetic PDF and have a 0.5B model identify it."""
    body = textwrap.dedent(
        """
        RFC 8259 The JavaScript Object Notation (JSON) Data Interchange Format
        Internet Engineering Task Force
        Abstract: JSON is a lightweight, text-based, language-independent data
        interchange format. It was derived from the JavaScript programming
        language. JSON defines a small set of formatting rules for the
        portable representation of structured data.
        """
    ).strip()
    pdf_bytes = _make_text_pdf(body)

    text, figures, *_ = extractor(pdf_bytes)
    assert "JSON" in text or "Object Notation" in text, (
        f"Extractor lost the body text. Got: {text[:200]!r}"
    )

    prompt = textwrap.dedent(
        f"""
        You read attached documents and answer in 1-2 sentences.

        [DOCUMENT]
        {text[:3000]}
        [/DOCUMENT]

        Question: Which RFC number does this document define and what is JSON?
        Answer:
        """
    ).strip()

    out = llama(
        prompt,
        max_tokens=160,
        temperature=0.2,
        stop=["\n\n", "</s>", "<|im_end|>"],
    )
    answer = out["choices"][0]["text"].strip().lower()
    print(f"\n[answer]\n{answer}\n")

    matched_keywords = [kw for kw in ("8259", "json", "object notation") if kw in answer]
    assert len(matched_keywords) >= 2, (
        f"Answer missed too many keywords. Got: {answer!r}; "
        f"matched: {matched_keywords}"
    )
