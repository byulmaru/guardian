import io
import unittest

from glaze_worker.storage import (
    InvalidObjectKey,
    ObjectReadError,
    ResultWriteError,
    S3ObjectStore,
    SourceObjectNotFound,
)


class ApiError(Exception):
    def __init__(self, code, status):
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class FakeBody(io.BytesIO):
    closed_by_store = False

    def close(self):
        self.closed_by_store = True
        super().close()


class FakeS3:
    def __init__(self):
        self.source = {"Body": FakeBody(b"source"), "ETag": '"source-etag"'}
        self.result = None
        self.put_error = None
        self.get_error = None
        self.head_error = None
        self.put_calls = []

    def get_object(self, **kwargs):
        if self.get_error:
            raise self.get_error
        return self.source

    def head_object(self, **kwargs):
        if self.head_error:
            raise self.head_error
        if self.result is None:
            raise ApiError("NotFound", 404)
        return self.result

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        if self.put_error:
            raise self.put_error
        self.result = {"Metadata": kwargs["Metadata"]}


class S3ObjectStoreTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeS3()
        self.store = S3ObjectStore(self.client, "bucket")

    def test_read_source_returns_bytes_and_etag_and_closes_body(self):
        source = self.store.read_source("uploads/photo.webp")

        self.assertEqual(source.data, b"source")
        self.assertEqual(source.etag, '"source-etag"')
        self.assertTrue(self.client.source["Body"].closed_by_store)

    def test_read_missing_source_is_typed(self):
        self.client.get_error = ApiError("NoSuchKey", 404)

        with self.assertRaises(SourceObjectNotFound):
            self.store.read_source("uploads/missing.webp")

    def test_missing_result_metadata_returns_none(self):
        self.assertIsNone(self.store.result_metadata("results/photo.webp"))

    def test_existing_result_metadata_is_normalized(self):
        self.client.result = {"Metadata": {"Glaze-Fingerprint": "abc"}}

        self.assertEqual(self.store.result_metadata("results/photo.webp"), {"glaze-fingerprint": "abc"})

    def test_write_uses_atomic_create_and_records_fingerprint(self):
        written = self.store.put_result_if_absent("results/photo.webp", b"protected", "abc")

        self.assertTrue(written)
        call = self.client.put_calls[0]
        self.assertEqual(call["IfNoneMatch"], "*")
        self.assertEqual(call["Metadata"], {"glaze-fingerprint": "abc"})
        self.assertEqual(call["ContentType"], "image/webp")

    def test_conditional_write_race_is_reported_to_caller(self):
        self.client.put_error = ApiError("PreconditionFailed", 412)

        self.assertFalse(self.store.put_result_if_absent("results/photo.webp", b"protected", "abc"))

    def test_storage_api_failures_are_typed_without_provider_details(self):
        self.client.head_error = ApiError("AccessDenied", 403)
        with self.assertRaisesRegex(ObjectReadError, "object storage lookup failed"):
            self.store.result_metadata("results/photo.webp")

        self.client.put_error = ApiError("InternalError", 500)
        with self.assertRaisesRegex(ResultWriteError, "object storage write failed"):
            self.store.put_result_if_absent("results/photo.webp", b"protected", "abc")

    def test_rejects_invalid_s3_keys(self):
        for key in ("", "  ", None, "x" * 1025, "bad\nkey"):
            with self.subTest(key=key):
                with self.assertRaises(InvalidObjectKey):
                    self.store.read_source(key)


if __name__ == "__main__":
    unittest.main()
