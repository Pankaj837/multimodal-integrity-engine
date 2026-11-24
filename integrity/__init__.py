"""Multimodal submission integrity and plagiarism detection engine.

Pipeline stages (one module each):

    extractors/   PDF, DOCX and audio/video (Whisper ASR) -> text blocks
    normalize     Unicode / whitespace / hyphenation clean-up, header-footer removal
    segmentation  blocks -> located sentences (page, paragraph or timestamp)
    chunker       sentences -> ~200-token overlapping chunks
    embedder      chunks and sentences -> L2-normalised MiniLM vectors
    indexer       FAISS inner-product index with same-document exclusion
    boilerplate   shared-template detection (judgment scenario)
    matching      peer / cross-modal / external passage matching
    ai_detect     AI-generated-text likelihood (perplexity, burstiness, classifier)
    scoring       percentages and risk level
    reporter      per-document JSON + Markdown reports, batch summary
    pipeline      orchestration used by run_integrity_check.py
"""

__version__ = "1.0.0"
