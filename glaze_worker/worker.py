"""Standalone Temporal worker for Glaze protection activities."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Callable, Mapping

from .activity import ActivityRuntime, configure_activity, protect_image
from .storage import S3ObjectStore


DEFAULT_TASK_QUEUE = "glaze-protect"


class WorkerStartupError(RuntimeError):
    """A required worker dependency or configuration is not available."""


@dataclass(frozen=True)
class WorkerConfig:
    s3_endpoint: str
    s3_region: str
    s3_bucket: str
    model_dir: Path
    device: str
    alexnet_weights: Path
    temporal_address: str
    temporal_namespace: str
    task_queue: str


def load_config(environ: Mapping[str, str] | None = None) -> WorkerConfig:
    env = os.environ if environ is None else environ
    endpoint = _required(env, "GLAZE_S3_ENDPOINT")
    region = env.get("GLAZE_S3_REGION", "").strip() or "auto"
    bucket = _required(env, "GLAZE_S3_BUCKET")
    model_dir = Path(_required(env, "GLAZE_MODEL_DIR")).expanduser()
    device = _required(env, "GLAZE_DEVICE")
    alexnet_weights = Path(_required(env, "GLAZE_ALEXNET_WEIGHTS")).expanduser()
    temporal_address = _required(env, "TEMPORAL_ADDRESS")
    temporal_namespace = _required(env, "TEMPORAL_NAMESPACE")
    task_queue = env.get("GLAZE_TASK_QUEUE", "").strip() or DEFAULT_TASK_QUEUE

    parsed_endpoint = urlparse(endpoint)
    if (
        parsed_endpoint.scheme not in {"http", "https"}
        or not parsed_endpoint.hostname
        or parsed_endpoint.username is not None
        or parsed_endpoint.password is not None
        or parsed_endpoint.query
        or parsed_endpoint.fragment
    ):
        raise WorkerStartupError("GLAZE_S3_ENDPOINT must be an HTTP(S) endpoint")
    if not model_dir.is_dir():
        raise WorkerStartupError("GLAZE_MODEL_DIR must point to an existing directory")
    if not alexnet_weights.is_file():
        raise WorkerStartupError("GLAZE_ALEXNET_WEIGHTS must point to an existing file")

    return WorkerConfig(
        s3_endpoint=endpoint,
        s3_region=region,
        s3_bucket=bucket,
        model_dir=model_dir,
        device=device,
        alexnet_weights=alexnet_weights,
        temporal_address=temporal_address,
        temporal_namespace=temporal_namespace,
        task_queue=task_queue,
    )


def create_object_store(
    config: WorkerConfig,
    client_factory: Callable[..., Any] | None = None,
) -> S3ObjectStore:
    if client_factory is None:
        try:
            import boto3
        except ModuleNotFoundError as exc:
            if exc.name != "boto3":
                raise
            raise WorkerStartupError("boto3 is required for the object store") from None
        client_factory = boto3.client
    try:
        client = client_factory(
            "s3", endpoint_url=config.s3_endpoint, region_name=config.s3_region
        )
        client.head_bucket(Bucket=config.s3_bucket)
    except Exception:
        raise WorkerStartupError("object storage readiness check failed") from None
    return S3ObjectStore(client, config.s3_bucket)


def load_engine(
    config: WorkerConfig,
    engine_factory: Callable[..., Any] | None = None,
) -> Any:
    if engine_factory is None:
        try:
            from .inference import ProtectionEngine
        except ModuleNotFoundError as exc:
            if exc.name != "glaze_worker.inference":
                raise
            raise WorkerStartupError("protection engine is unavailable") from None
        engine_factory = ProtectionEngine
    try:
        engine = engine_factory(
            str(config.model_dir),
            config.device,
            str(config.alexnet_weights),
        )
        identity = getattr(engine, "identity", None)
        if not isinstance(identity, str) or not identity:
            raise WorkerStartupError("protection engine identity is unavailable")
    except WorkerStartupError:
        raise
    except Exception:
        raise WorkerStartupError("protection engine initialization failed") from None
    return engine


async def connect_temporal(
    config: WorkerConfig,
    client_factory: Callable[..., Any] | None = None,
) -> Any:
    if client_factory is None:
        try:
            from temporalio.client import Client
        except ModuleNotFoundError as exc:
            if exc.name != "temporalio":
                raise
            raise WorkerStartupError("temporalio is required for the worker") from None
        client_factory = Client.connect
    try:
        return await client_factory(
            config.temporal_address,
            namespace=config.temporal_namespace,
        )
    except Exception:
        raise WorkerStartupError("Temporal client connection failed") from None


async def run_worker() -> None:
    config = load_config()
    store = create_object_store(config)
    engine = load_engine(config)
    configure_activity(ActivityRuntime(engine, store))
    client = await connect_temporal(config)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="glaze-activity") as activity_executor:
        from temporalio.worker import Worker

        worker = Worker(
            client,
            task_queue=config.task_queue,
            activities=[protect_image],
            workflows=[],
            activity_executor=activity_executor,
            max_concurrent_activities=1,
        )
        await worker.run()


def _required(environ: Mapping[str, str], key: str) -> str:
    value = environ.get(key, "").strip()
    if not value:
        raise WorkerStartupError(f"{key} must be configured")
    return value


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
