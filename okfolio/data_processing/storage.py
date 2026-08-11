"""Asset storage adapters, including the user-provided S3Writer contract."""
from __future__ import annotations

import hashlib
import importlib
import mimetypes
import os
from pathlib import Path
from typing import Any, Mapping, Protocol


class AssetWriter(Protocol):
    def write(self, key: str, data: bytes, *, content_type: str) -> str: ...


class LocalAssetWriter:
    def __init__(self, root: Path, *, uri_prefix: str = "images"):
        self.root = root.resolve()
        self.uri_prefix = uri_prefix.strip("/")

    def write(self, key: str, data: bytes, *, content_type: str) -> str:
        del content_type
        relative = Path(key)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("asset key must be a safe relative path")
        target = (self.root / relative).resolve()
        target.relative_to(self.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return f"{self.uri_prefix}/{relative.as_posix()}"


class S3WriterAssetWriter:
    """Adapt an object exposing ``write(key, bytes)`` to the AssetWriter API."""

    def __init__(
        self,
        writer: Any,
        *,
        bucket: str,
        prefix: str = "",
    ):
        if not bucket.strip():
            raise ValueError("S3 bucket must not be empty")
        if not callable(getattr(writer, "write", None)):
            raise TypeError("S3Writer object must expose write(key, bytes)")
        self.writer = writer
        self.bucket = bucket.strip()
        self.prefix = prefix.strip("/")

    def write(self, key: str, data: bytes, *, content_type: str) -> str:
        del content_type
        relative = Path(key)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("asset key must be a safe relative path")
        object_key = "/".join(
            part
            for part in (self.prefix, relative.as_posix())
            if part
        )
        self.writer.write(object_key, data)
        return f"s3://{self.bucket}/{object_key}"

    @classmethod
    def from_factory(
        cls,
        factory_spec: str,
        *,
        bucket: str,
        prefix: str = "",
        environ: Mapping[str, str] | None = None,
    ) -> "S3WriterAssetWriter":
        """Load ``package.module:function`` without embedding credentials."""
        if ":" not in factory_spec:
            raise ValueError("S3 writer factory must be package.module:function")
        module_name, function_name = factory_spec.split(":", 1)
        factory = getattr(importlib.import_module(module_name), function_name)
        writer = factory(dict(os.environ if environ is None else environ))
        return cls(writer, bucket=bucket, prefix=prefix)


def asset_key(document_id: str, source: Path) -> str:
    suffix = source.suffix.lower() or ".bin"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return f"{document_id}/{digest[:24]}{suffix}"


def content_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
