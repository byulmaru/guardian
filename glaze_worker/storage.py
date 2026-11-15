"""S3-compatible object storage boundary for Glaze activities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ObjectStoreError(RuntimeError):
    """Base class for safe, user-independent object-store failures."""


class InvalidObjectKey(ValueError):
    """An object key cannot be represented safely by the S3 API."""


class SourceObjectNotFound(ObjectStoreError):
    """The requested source key does not exist."""


class ObjectReadError(ObjectStoreError):
    """The object store could not read an object."""


class ResultWriteError(ObjectStoreError):
    """The object store could not write a result object."""


@dataclass(frozen=True)
class SourceObject:
    data: bytes
    etag: str


def validate_object_key(key: object) -> str:
    if not isinstance(key, str) or not key.strip():
        raise InvalidObjectKey("object key must be a non-empty string")
    try:
        encoded = key.encode("utf-8")
    except UnicodeEncodeError:
        raise InvalidObjectKey("object key is not valid UTF-8") from None
    if len(encoded) > 1024 or any(ord(char) < 32 or ord(char) == 127 for char in key):
        raise InvalidObjectKey("object key is not valid for S3")
    return key


class S3ObjectStore:
    """Small injectable adapter around the S3 API."""

    def __init__(self, client: Any, bucket: str):
        if not isinstance(bucket, str) or not bucket.strip():
            raise ValueError("object-store bucket must be configured")
        self._client = client
        self._bucket = bucket

    def read_source(self, key: str) -> SourceObject:
        key = validate_object_key(key)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            if _is_not_found(exc):
                raise SourceObjectNotFound("source object was not found") from None
            raise ObjectReadError("object storage read failed") from None

        body = response.get("Body")
        try:
            if body is None or not callable(getattr(body, "read", None)):
                raise ObjectReadError("object storage returned an invalid source")
            data = body.read()
            etag = response.get("ETag")
            if not isinstance(data, bytes) or not isinstance(etag, str) or not etag:
                raise ObjectReadError("object storage returned an invalid source")
            return SourceObject(data=data, etag=etag)
        except ObjectStoreError:
            raise
        except Exception:
            raise ObjectReadError("object storage read failed") from None
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def result_metadata(self, key: str) -> dict[str, str] | None:
        key = validate_object_key(key)
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise ObjectReadError("object storage lookup failed") from None
        metadata = response.get("Metadata")
        if not isinstance(metadata, dict):
            return {}
        return {name.lower(): value for name, value in metadata.items() if isinstance(name, str) and isinstance(value, str)}

    def put_result_if_absent(self, key: str, data: bytes, fingerprint: str) -> bool:
        key = validate_object_key(key)
        if not isinstance(data, bytes):
            raise TypeError("result data must be bytes")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("result fingerprint must be set")
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType="image/webp",
                Metadata={"glaze-fingerprint": fingerprint},
                IfNoneMatch="*",
            )
        except Exception as exc:
            if _is_precondition_failure(exc):
                return False
            raise ResultWriteError("object storage write failed") from None
        return True


def _error_details(exc: BaseException) -> tuple[str | None, int | None]:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None, None
    error = response.get("Error")
    code = error.get("Code") if isinstance(error, dict) else None
    metadata = response.get("ResponseMetadata")
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    return (code if isinstance(code, str) else None, status if isinstance(status, int) else None)


def _is_not_found(exc: BaseException) -> bool:
    code, status = _error_details(exc)
    return code in {"NoSuchKey", "NotFound"} or status == 404


def _is_precondition_failure(exc: BaseException) -> bool:
    code, status = _error_details(exc)
    return code in {"PreconditionFailed", "ConditionalRequestConflict"} or status in {409, 412}
