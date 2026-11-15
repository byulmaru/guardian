"""Temporal activity and idempotent object-store processing."""

from __future__ import annotations

import contextvars
import hashlib
import json
import queue
import threading
from typing import Any, Protocol

from .inference import InvalidImageError, InvalidSettingsError, ProtectionSettings
from .storage import (
    InvalidObjectKey,
    ObjectStoreError,
    ResultWriteError,
    S3ObjectStore,
    SourceObjectNotFound,
    validate_object_key,
)

try:
    from temporalio import activity as temporal_activity
    from temporalio.exceptions import ApplicationError, CancelledError
except ModuleNotFoundError as exc:
    if exc.name != "temporalio":
        raise
    temporal_activity = None
    ApplicationError = None
    CancelledError = None


HEARTBEAT_INTERVAL_SECONDS = 10
FINGERPRINT_METADATA_KEY = "glaze-fingerprint"


class ActivityInputError(ValueError):
    pass


class ActivityConflict(RuntimeError):
    pass


class ActivityCancelled(Exception):
    pass


class ActivityExecutionError(RuntimeError):
    pass


class ActivityContext(Protocol):
    def heartbeat(self, details: dict[str, int]) -> None: ...

    def is_cancelled(self) -> bool: ...

    def is_cancellation_error(self, error: BaseException) -> bool: ...


class _TemporalActivityContext:
    def heartbeat(self, details: dict[str, int]) -> None:
        temporal_activity.heartbeat(details)

    def is_cancelled(self) -> bool:
        return temporal_activity.is_cancelled()

    def is_cancellation_error(self, error: BaseException) -> bool:
        return isinstance(error, CancelledError)


def request_fingerprint(
    source_key: str,
    source_etag: str,
    engine_identity: str,
    settings: ProtectionSettings,
) -> str:
    encoded = json.dumps(
        {
            "engine_identity": engine_identity,
            "settings": settings.as_dict(),
            "source_etag": source_etag,
            "source_key": source_key,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ActivityRuntime:
    def __init__(self, engine: Any, store: S3ObjectStore, heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS):
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat interval must be positive")
        self._engine = engine
        self._store = store
        self._heartbeat_interval = heartbeat_interval

    def execute(self, request: object, activity_context: ActivityContext) -> dict[str, str]:
        source_key, result_key, settings = _parse_request(request)
        engine_identity = getattr(self._engine, "identity", None)
        if not isinstance(engine_identity, str) or not engine_identity:
            raise ActivityExecutionError("protection engine identity is unavailable")

        stopped = threading.Event()
        cancelled = threading.Event()
        progress_lock = threading.Lock()
        progress = {"completed_steps": 0, "total_steps": 0}
        heartbeat_errors: queue.SimpleQueue[BaseException] = queue.SimpleQueue()

        def heartbeat_loop() -> None:
            while not stopped.is_set():
                try:
                    with progress_lock:
                        details = dict(progress)
                    activity_context.heartbeat(details)
                    if activity_context.is_cancelled():
                        cancelled.set()
                        return
                except Exception as error:
                    if activity_context.is_cancellation_error(error):
                        cancelled.set()
                    else:
                        heartbeat_errors.put(error)
                    return
                if stopped.wait(self._heartbeat_interval):
                    return

        context = contextvars.copy_context()
        heartbeat_thread = threading.Thread(
            target=context.run,
            args=(heartbeat_loop,),
            name="glaze-activity-heartbeat",
            daemon=True,
        )

        def check_activity() -> None:
            if cancelled.is_set():
                raise ActivityCancelled()
            try:
                if activity_context.is_cancelled():
                    cancelled.set()
                    raise ActivityCancelled()
            except ActivityCancelled:
                raise
            except Exception as error:
                if activity_context.is_cancellation_error(error):
                    cancelled.set()
                    raise ActivityCancelled() from None
                raise ActivityExecutionError("activity status check failed") from None
            try:
                heartbeat_errors.get_nowait()
            except queue.Empty:
                return
            raise ActivityExecutionError("activity heartbeat failed")

        def report_progress(completed_steps: int, total_steps: int) -> None:
            if type(completed_steps) is not int or type(total_steps) is not int:
                raise ActivityExecutionError("protection engine returned invalid progress")
            if total_steps <= 0 or completed_steps < 0 or completed_steps > total_steps:
                raise ActivityExecutionError("protection engine returned invalid progress")
            check_activity()
            with progress_lock:
                progress["completed_steps"] = completed_steps
                progress["total_steps"] = total_steps

        heartbeat_thread.start()
        try:
            check_activity()
            source = self._store.read_source(source_key)
            fingerprint = request_fingerprint(source_key, source.etag, engine_identity, settings)

            metadata = self._store.result_metadata(result_key)
            if metadata is not None:
                if metadata.get(FINGERPRINT_METADATA_KEY) == fingerprint:
                    return {"result_key": result_key}
                raise ActivityConflict("result key already contains a different image")

            try:
                result = self._engine.protect(source.data, settings, report_progress)
            except InvalidImageError:
                raise
            except (ActivityCancelled, ActivityExecutionError):
                raise
            except Exception as error:
                if activity_context.is_cancellation_error(error):
                    raise ActivityCancelled() from None
                raise ActivityExecutionError("image protection failed") from None
            check_activity()
            if not isinstance(result, bytes):
                raise ActivityExecutionError("protection engine returned invalid output")

            if self._store.put_result_if_absent(result_key, result, fingerprint):
                return {"result_key": result_key}

            metadata = self._store.result_metadata(result_key)
            if metadata is None:
                # Retry conditional write conflicts when no committed object is visible.
                raise ResultWriteError("conditional result write was not visible after the race")
            if metadata.get(FINGERPRINT_METADATA_KEY) == fingerprint:
                return {"result_key": result_key}
            raise ActivityConflict("result key already contains a different image")
        except (ActivityInputError, ActivityConflict, ActivityCancelled, ActivityExecutionError, ObjectStoreError):
            raise
        except InvalidObjectKey:
            raise ActivityInputError("invalid protection request") from None
        finally:
            stopped.set()
            heartbeat_thread.join()


def _parse_request(request: object) -> tuple[str, str, ProtectionSettings]:
    if not isinstance(request, dict) or set(request) != {"source_key", "result_key", "settings"}:
        raise ActivityInputError("invalid protection request")
    try:
        source_key = validate_object_key(request["source_key"])
        result_key = validate_object_key(request["result_key"])
    except InvalidObjectKey:
        raise ActivityInputError("invalid protection request") from None
    if source_key == result_key:
        raise ActivityInputError("source and result keys must differ")
    settings = ProtectionSettings.from_dict(request["settings"])
    return source_key, result_key, settings


_runtime: ActivityRuntime | None = None


def configure_activity(runtime: ActivityRuntime) -> None:
    global _runtime
    _runtime = runtime


def _activity_definition(fn):
    if temporal_activity is None:
        return fn
    return temporal_activity.defn(name="protect_image")(fn)


@_activity_definition
def protect_image(request: dict[str, Any]) -> dict[str, str]:
    if temporal_activity is None or ApplicationError is None:
        raise RuntimeError("Temporal Python SDK is required")
    if _runtime is None:
        raise ApplicationError("protection activity is not ready", type="ActivityNotReady")
    try:
        return _runtime.execute(request, _TemporalActivityContext())
    except ActivityInputError:
        raise ApplicationError(
            "invalid protection request",
            type="InvalidProtectionRequest",
            non_retryable=True,
        ) from None
    except InvalidSettingsError:
        raise ApplicationError(
            "invalid protection settings",
            type="InvalidProtectionSettings",
            non_retryable=True,
        ) from None
    except ActivityConflict:
        raise ApplicationError(
            "result key already contains a different image",
            type="ResultKeyConflict",
            non_retryable=True,
        ) from None
    except SourceObjectNotFound:
        raise ApplicationError("source object was not found", type="SourceObjectNotFound", non_retryable=True) from None
    except ActivityCancelled:
        raise CancelledError() from None
    except InvalidImageError:
        raise ApplicationError(
            "image input is invalid or unsupported",
            type="InvalidImage",
            non_retryable=True,
        ) from None
    except ObjectStoreError:
        raise ApplicationError("object storage operation failed", type="ObjectStorageFailure") from None
    except ActivityExecutionError:
        raise ApplicationError("image protection failed", type="ProtectionFailure") from None
