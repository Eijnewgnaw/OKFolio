"""PDF-to-Article processing primitives for OKFolio."""

from typing import Any

from .activation import activate_article, deactivate_article
from .mineru import MinerUContentAdapter
from .models import ArticleSegment, Block, DocumentIR, ProcessingResult
from .pdf_worker import PDFPageRenderer, parse_pdf_with_vlm
from .page_role import (
    OpenAICompatiblePageRoleClassifier,
    PageRoleResult,
)
from .pipeline import process_mineru_output, render_article
from .s3 import S3CompatibleAssetWriter
from .segmenter import segment_document
from .storage import LocalAssetWriter, S3WriterAssetWriter
from .structure import (
    OutlineEntry,
    PageDecision,
    StructurePolicy,
    StructureNormalization,
    document_from_dict,
    normalize_document_structure,
)
from .vlm import OpenAICompatiblePageParser, PageParseError, PageParseResult


def __getattr__(name: str) -> Any:
    """Load the optional official MinerU client only when it is requested."""
    if name == "OfficialMinerUPageParser":
        from .mineru_official import OfficialMinerUPageParser

        return OfficialMinerUPageParser
    raise AttributeError(name)

__all__ = [
    "ArticleSegment",
    "activate_article",
    "deactivate_article",
    "Block",
    "DocumentIR",
    "LocalAssetWriter",
    "MinerUContentAdapter",
    "OfficialMinerUPageParser",
    "OpenAICompatiblePageParser",
    "OpenAICompatiblePageRoleClassifier",
    "OutlineEntry",
    "PageDecision",
    "PDFPageRenderer",
    "PageParseError",
    "PageParseResult",
    "PageRoleResult",
    "ProcessingResult",
    "S3CompatibleAssetWriter",
    "S3WriterAssetWriter",
    "StructureNormalization",
    "StructurePolicy",
    "document_from_dict",
    "normalize_document_structure",
    "parse_pdf_with_vlm",
    "process_mineru_output",
    "render_article",
    "segment_document",
]
