import threading
import time
import unittest

from glaze_worker.activity import (
    ActivityCancelled,
    ActivityConflict,
    ActivityExecutionError,
    ActivityInputError,
    ActivityRuntime,
    configure_activity,
    protect_image,
    request_fingerprint,
)
from glaze_worker.inference import InvalidImageError, InvalidSettingsError, ProtectionSettings
from glaze_worker.storage import ObjectReadError, ResultWriteError, SourceObject

try:
    from temporalio.testing import ActivityEnvironment
    from temporalio.exceptions import ApplicationError
except ModuleNotFoundError as exc:
    if exc.name != "temporalio":
        raise
    ActivityEnvironment = None
    ApplicationError = None


class FakeContext:
    def __init__(self, cancel_after=None):
        self.heartbeats = []
        self.cancel_after = cancel_after
        self.cancelled = threading.Event()

    def heartbeat(self, details):
        self.heartbeats.append(details)
        if self.cancel_after and len(self.heartbeats) >= self.cancel_after:
            self.cancelled.set()

    def is_cancelled(self):
        return self.cancelled.is_set()

    def is_cancellation_error(self, error):
        return isinstance(error, ActivityCancelled)


class FakeEngine:
    identity = "sha256:stable-model-identity"

    def __init__(self, result=b"lossless-webp", wait_for=None, fail=None):
        self.result = result
        self.wait_for = wait_for
        self.fail = fail
        self.calls = 0
        self.input_data = None
        self.settings = None
        self.progress_values = []

    def protect(self, image_bytes, settings, progress=None):
        self.calls += 1
        self.input_data = image_bytes
        self.settings = settings
        if progress:
            progress(0, 1)
            self.progress_values.append((0, 1))
        if self.wait_for is not None:
            if not self.wait_for.wait(timeout=2):
                raise AssertionError("expected event was not signaled")
        if self.fail is not None:
            raise self.fail
        if progress:
            progress(1, 1)
            self.progress_values.append((1, 1))
        return self.result


class FakeStore:
    def __init__(self):
        self.source = SourceObject(b"original", '"etag-1"')
        self.metadata = None
        self.write_result = True
        self.write_race_metadata = None
        self.write_calls = []
        self.read_error = None
        self.source_reads = []
        self.metadata_reads = []

    def read_source(self, key):
        self.source_reads.append(key)
        if self.read_error:
            raise self.read_error
        return self.source

    def result_metadata(self, key):
        self.metadata_reads.append(key)
        return self.metadata

    def put_result_if_absent(self, key, data, fingerprint):
        self.write_calls.append((key, data, fingerprint))
        if not self.write_result:
            self.metadata = self.write_race_metadata
            return False
        self.metadata = {"glaze-fingerprint": fingerprint}
        return True


class ActivityRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore()
        self.engine = FakeEngine()
        self.runtime = ActivityRuntime(self.engine, self.store, heartbeat_interval=0.005)
        self.request = {
            "source_key": "uploads/input.webp",
            "result_key": "results/output.webp",
            "settings": {},
        }

    def fingerprint(self, settings=None):
        normalized = ProtectionSettings.from_dict({} if settings is None else settings)
        return request_fingerprint(
            "uploads/input.webp", '"etag-1"', self.engine.identity, normalized
        )

    def test_valid_request_protects_and_returns_result_key(self):
        context = FakeContext()
        self.engine.wait_for = threading.Event()
        self.engine.wait_for.set()

        result = self.runtime.execute(self.request, context)

        self.assertEqual(result, {"result_key": "results/output.webp"})
        self.assertEqual(self.engine.input_data, b"original")
        self.assertEqual(len(self.store.write_calls), 1)
        self.assertEqual(self.store.write_calls[0][1], b"lossless-webp")
        self.assertEqual(self.store.write_calls[0][2], self.fingerprint())
        self.assertTrue(context.heartbeats)
        self.assertTrue(all(set(item) == {"completed_steps", "total_steps"} for item in context.heartbeats))
        heartbeat_count = len(context.heartbeats)
        time.sleep(0.02)
        self.assertEqual(len(context.heartbeats), heartbeat_count, "heartbeat helper should stop before returning")

    def test_existing_matching_fingerprint_reuses_result_without_processing(self):
        self.store.metadata = {
            "glaze-fingerprint": self.fingerprint()
        }

        result = self.runtime.execute(self.request, FakeContext())

        self.assertEqual(result, {"result_key": "results/output.webp"})
        self.assertEqual(self.engine.calls, 0)
        self.assertEqual(self.store.write_calls, [])

    def test_existing_different_fingerprint_is_non_retryable_conflict(self):
        self.store.metadata = {"glaze-fingerprint": "different"}

        with self.assertRaises(ActivityConflict):
            self.runtime.execute(self.request, FakeContext())

        self.assertEqual(self.engine.calls, 0)

    def test_conditional_write_race_reuses_only_matching_result(self):
        fingerprint = self.fingerprint()
        self.store.write_result = False
        self.store.write_race_metadata = {"glaze-fingerprint": fingerprint}

        result = self.runtime.execute(self.request, FakeContext())

        self.assertEqual(result, {"result_key": "results/output.webp"})
        self.assertEqual(self.engine.calls, 1)

    def test_conditional_write_race_with_different_metadata_fails(self):
        self.store.write_result = False
        self.store.write_race_metadata = {"glaze-fingerprint": "different"}

        with self.assertRaises(ActivityConflict):
            self.runtime.execute(self.request, FakeContext())

    def test_runtime_conditional_write_race_without_visible_object_raises_storage_error(self):
        self.store.write_result = False
        self.store.write_race_metadata = None

        with self.assertRaises(ResultWriteError):
            self.runtime.execute(self.request, FakeContext())

    def test_invalid_payloads_and_same_keys_are_rejected(self):
        invalid = [
            [],
            {"source_key": "only-source"},
            {"source_key": "a", "result_key": "b"},
            {"source_key": "a", "result_key": "b", "extra": "field"},
            {"source_key": "", "result_key": "b"},
            {"source_key": "same", "result_key": "same"},
        ]
        for request in invalid:
            with self.subTest(request=request):
                with self.assertRaises(ActivityInputError):
                    self.runtime.execute(request, FakeContext())

    def test_invalid_settings_are_rejected_before_storage(self):
        request = {**self.request, "settings": {"target_seed": "private prompt value"}}

        with self.assertRaises(InvalidSettingsError):
            self.runtime.execute(request, FakeContext())

        self.assertEqual(self.store.source_reads, [])
        self.assertEqual(self.store.metadata_reads, [])
        self.assertEqual(self.store.write_calls, [])

    def test_engine_failure_is_typed_and_does_not_write(self):
        self.engine.fail = RuntimeError("private provider detail")

        with self.assertRaisesRegex(ActivityExecutionError, "image protection failed"):
            self.runtime.execute(self.request, FakeContext())

        self.assertEqual(self.store.write_calls, [])

    def test_storage_failure_is_preserved_as_typed_error(self):
        self.store.read_error = ObjectReadError("object storage read failed")

        with self.assertRaises(ObjectReadError):
            self.runtime.execute(self.request, FakeContext())

    def test_cancellation_stops_processing_and_cleans_heartbeat_thread(self):
        context = FakeContext(cancel_after=1)
        self.engine.wait_for = context.cancelled

        with self.assertRaises(ActivityCancelled):
            self.runtime.execute(self.request, context)

        self.assertEqual(self.store.write_calls, [])
        heartbeat_count = len(context.heartbeats)
        time.sleep(0.02)
        self.assertEqual(len(context.heartbeats), heartbeat_count)

    def test_fingerprint_changes_with_source_or_engine_identity(self):
        original = request_fingerprint("source", "etag", "engine-a", ProtectionSettings.from_dict({}))

        self.assertNotEqual(
            original,
            request_fingerprint("other-source", "etag", "engine-a", ProtectionSettings.from_dict({})),
        )
        self.assertNotEqual(
            original,
            request_fingerprint("source", "other-etag", "engine-a", ProtectionSettings.from_dict({})),
        )
        self.assertNotEqual(
            original,
            request_fingerprint("source", "etag", "engine-b", ProtectionSettings.from_dict({})),
        )

    def test_omitted_setting_fields_match_explicit_defaults(self):
        defaults = ProtectionSettings.from_dict({})
        explicit_request = {**self.request, "settings": defaults.as_dict()}

        self.assertEqual(self.fingerprint(), self.fingerprint(explicit_request["settings"]))
        self.runtime.execute(self.request, FakeContext())
        result = self.runtime.execute(explicit_request, FakeContext())

        self.assertEqual(result, {"result_key": "results/output.webp"})
        self.assertEqual(self.engine.calls, 1)
        self.assertEqual(len(self.store.write_calls), 1)

    def test_full_normalized_settings_are_forwarded_to_engine(self):
        values = {
            "max_delta": 9,
            "iterations": 400,
            "learning_rate": 0.02,
            "lpips_threshold": 0.15,
            "lpips_weight": 5.0,
            "target_prompt": "a private test prompt",
            "target_seed": 7,
            "target_steps": 15,
            "target_strength": 0.6,
            "target_guidance": 8.0,
        }
        expected = ProtectionSettings.from_dict(values)
        request = {**self.request, "settings": values}

        self.runtime.execute(request, FakeContext())

        self.assertEqual(self.engine.settings.as_dict(), expected.as_dict())

    def test_each_target_and_delta_setting_conflicts_with_existing_result(self):
        self.store.metadata = {"glaze-fingerprint": self.fingerprint()}
        changed_settings = [
            {"target_seed": 17},
            {"target_prompt": "different prompt"},
            {"target_steps": 31},
            {"max_delta": 7},
        ]

        for settings in changed_settings:
            with self.subTest(settings=settings):
                request = {**self.request, "settings": settings}
                with self.assertRaises(ActivityConflict):
                    self.runtime.execute(request, FakeContext())

        self.assertEqual(self.engine.calls, 0)
        self.assertEqual(self.store.write_calls, [])

    @unittest.skipIf(ActivityEnvironment is None, "Temporal Python SDK is not installed")
    def test_invalid_settings_are_non_retryable_redacted_and_rejected_before_storage(self):
        configure_activity(self.runtime)
        request = {**self.request, "settings": {"target_seed": "private prompt value"}}

        with self.assertRaises(ApplicationError) as raised:
            ActivityEnvironment().run(protect_image, request)

        self.assertEqual(raised.exception.type, "InvalidProtectionSettings")
        self.assertTrue(raised.exception.non_retryable)
        self.assertEqual(raised.exception.message, "invalid protection settings")
        self.assertNotIn("private prompt value", raised.exception.message)
        self.assertEqual(self.store.source_reads, [])
        self.assertEqual(self.store.metadata_reads, [])
        self.assertEqual(self.store.write_calls, [])

    @unittest.skipIf(ActivityEnvironment is None, "Temporal Python SDK is not installed")
    def test_temporal_activity_environment_runs_without_a_server(self):
        context = ActivityEnvironment()
        heartbeats = []
        context.on_heartbeat = lambda *details: heartbeats.append(details)
        configure_activity(self.runtime)

        result = context.run(protect_image, self.request)

        self.assertEqual(result, {"result_key": "results/output.webp"})
        self.assertTrue(heartbeats)

    @unittest.skipIf(ActivityEnvironment is None, "Temporal Python SDK is not installed")
    def test_oversized_image_error_is_non_retryable_and_redacted(self):
        self.engine.fail = InvalidImageError("width 1025 exceeds 1024; private-object-key")
        configure_activity(self.runtime)

        with self.assertRaises(ApplicationError) as raised:
            ActivityEnvironment().run(protect_image, self.request)

        self.assertEqual(raised.exception.type, "InvalidImage")
        self.assertTrue(raised.exception.non_retryable)
        self.assertEqual(raised.exception.message, "image input is invalid or unsupported")
        self.assertNotIn("1025", raised.exception.message)
        self.assertNotIn("private-object-key", raised.exception.message)
        self.assertEqual(self.store.write_calls, [])

    @unittest.skipIf(ActivityEnvironment is None, "Temporal Python SDK is not installed")
    def test_generic_engine_value_error_remains_retryable(self):
        self.engine.fail = ValueError("private model failure")
        configure_activity(self.runtime)

        with self.assertRaises(ApplicationError) as raised:
            ActivityEnvironment().run(protect_image, self.request)

        self.assertEqual(raised.exception.type, "ProtectionFailure")
        self.assertFalse(raised.exception.non_retryable)
        self.assertEqual(raised.exception.message, "image protection failed")

    @unittest.skipIf(ActivityEnvironment is None, "Temporal Python SDK is not installed")
    def test_activity_conditional_write_race_without_visible_object_is_retryable(self):
        self.store.write_result = False
        self.store.write_race_metadata = None
        configure_activity(self.runtime)

        with self.assertRaises(ApplicationError) as raised:
            ActivityEnvironment().run(protect_image, self.request)

        self.assertEqual(raised.exception.type, "ObjectStorageFailure")
        self.assertFalse(raised.exception.non_retryable)


if __name__ == "__main__":
    unittest.main()
