"""Dependency-light S3-compatible asset writer for internal MinIO."""
from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx


def _hmac(key: bytes, value: str) -> bytes:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()


class S3CompatibleAssetWriter:
    """Upload assets to path-style S3/MinIO using AWS Signature Version 4."""

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
        timeout: float = 60.0,
        verify_tls: bool = True,
        public_base_url: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("S3 endpoint must be an http(s) URL")
        if not access_key.strip() or not secret_key.strip():
            raise ValueError("S3 credentials must not be empty")
        if not bucket.strip() or "/" in bucket:
            raise ValueError("S3 bucket must be a non-empty bucket name")
        self.endpoint = endpoint.rstrip("/")
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket.strip()
        self.prefix = prefix.strip("/")
        self.region = region
        self.timeout = timeout
        self.verify_tls = verify_tls
        self.public_base_url = public_base_url.rstrip("/")
        self.transport = transport
        self._clock_offset_seconds: float | None = None

    def write(self, key: str, data: bytes, *, content_type: str) -> str:
        relative = Path(key)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("asset key must be a safe relative path")
        object_key = "/".join(
            part for part in (self.prefix, relative.as_posix()) if part
        )
        response = self._request(
            "PUT",
            object_key=object_key,
            data=data,
            content_type=content_type,
        )
        if response.status_code not in {200, 201, 204}:
            message = response.text[:500].replace("\n", " ")
            raise RuntimeError(
                f"S3 upload failed: HTTP {response.status_code} {message}"
            )
        if self.public_base_url:
            return (
                f"{self.public_base_url}/{quote(self.bucket)}/"
                f"{quote(object_key, safe='/')}"
            )
        return f"s3://{self.bucket}/{object_key}"

    def bucket_exists(self) -> bool:
        response = self._request(
            "HEAD",
            object_key=None,
            data=b"",
            content_type="application/octet-stream",
        )
        if response.status_code in {200, 204}:
            return True
        if response.status_code == 404:
            return False
        raise RuntimeError(f"S3 bucket check failed: HTTP {response.status_code}")

    def create_bucket(self) -> None:
        response = self._request(
            "PUT",
            object_key=None,
            data=b"",
            content_type="application/octet-stream",
        )
        if response.status_code not in {200, 201, 204, 409}:
            raise RuntimeError(
                f"S3 bucket creation failed: HTTP {response.status_code}"
            )

    def _request(
        self,
        method: str,
        *,
        object_key: str | None,
        data: bytes,
        content_type: str,
        allow_clock_retry: bool = True,
    ) -> httpx.Response:
        parsed = urlsplit(self.endpoint)
        base_path = parsed.path.rstrip("/")
        resource = f"{base_path}/{quote(self.bucket, safe='-_.~')}"
        if object_key is not None:
            resource += "/" + quote(object_key, safe="/-_.~")
        canonical_uri = resource or "/"
        now = datetime.now(timezone.utc) + timedelta(
            seconds=self._clock_offset_seconds or 0.0
        )
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        payload_hash = hashlib.sha256(data).hexdigest()
        host = parsed.netloc
        headers = {
            "content-type": content_type,
            "host": host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
        }
        signed_headers = ";".join(sorted(headers))
        canonical_headers = "".join(
            f"{name}:{headers[name].strip()}\n" for name in sorted(headers)
        )
        canonical_request = "\n".join(
            [
                method,
                canonical_uri,
                "",
                canonical_headers,
                signed_headers,
                payload_hash,
            ]
        )
        scope = f"{date_stamp}/{self.region}/s3/aws4_request"
        string_to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                amz_date,
                scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            ]
        )
        date_key = _hmac(
            ("AWS4" + self.secret_key).encode("utf-8"),
            date_stamp,
        )
        region_key = _hmac(date_key, self.region)
        service_key = _hmac(region_key, "s3")
        signing_key = _hmac(service_key, "aws4_request")
        signature = hmac.new(
            signing_key,
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        authorization = (
            "AWS4-HMAC-SHA256 "
            f"Credential={self.access_key}/{scope},"
            f"SignedHeaders={signed_headers},"
            f"Signature={signature}"
        )
        request_headers = {
            "Content-Type": content_type,
            "Host": host,
            "X-Amz-Content-Sha256": payload_hash,
            "X-Amz-Date": amz_date,
            "Authorization": authorization,
        }
        with httpx.Client(
            timeout=self.timeout,
            trust_env=False,
            verify=self.verify_tls,
            transport=self.transport,
        ) as client:
            response = client.request(
                method,
                f"{parsed.scheme}://{host}{canonical_uri}",
                headers=request_headers,
                content=data,
            )
        if (
            allow_clock_retry
            and response.status_code == 403
            and self._clock_offset_seconds is None
            and self._synchronize_clock()
        ):
            return self._request(
                method,
                object_key=object_key,
                data=data,
                content_type=content_type,
                allow_clock_retry=False,
            )
        return response

    def _synchronize_clock(self) -> bool:
        """Use MinIO's HTTP Date without changing the worker or host clock."""
        parsed = urlsplit(self.endpoint)
        health_path = parsed.path.rstrip("/") + "/minio/health/live"
        try:
            with httpx.Client(
                timeout=self.timeout,
                trust_env=False,
                verify=self.verify_tls,
                transport=self.transport,
            ) as client:
                response = client.get(
                    f"{parsed.scheme}://{parsed.netloc}{health_path}"
                )
            server_date = response.headers.get("Date", "")
            server_now = parsedate_to_datetime(server_date)
            if server_now.tzinfo is None:
                server_now = server_now.replace(tzinfo=timezone.utc)
            self._clock_offset_seconds = (
                server_now.astimezone(timezone.utc) - datetime.now(timezone.utc)
            ).total_seconds()
            return True
        except (OSError, TypeError, ValueError, httpx.HTTPError):
            return False
