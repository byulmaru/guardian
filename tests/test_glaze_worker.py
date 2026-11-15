import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from glaze_worker.worker import create_object_store, load_config


class WorkerConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.model_dir = root / "models"
        self.model_dir.mkdir()
        self.alexnet_weights = root / "alexnet.pth"
        self.alexnet_weights.touch()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _environment(self, **extra):
        environment = {
            "GLAZE_S3_ENDPOINT": "http://localhost:9000",
            "GLAZE_S3_BUCKET": "glaze-test",
            "GLAZE_MODEL_DIR": str(self.model_dir),
            "GLAZE_DEVICE": "cpu",
            "GLAZE_ALEXNET_WEIGHTS": str(self.alexnet_weights),
            "TEMPORAL_ADDRESS": "localhost:7233",
            "TEMPORAL_NAMESPACE": "default",
        }
        environment.update(extra)
        return environment

    def _assert_region_and_readiness(self, environment, expected_region):
        config = load_config(environment)
        client = Mock()
        client_factory = Mock(return_value=client)

        create_object_store(config, client_factory)

        client_factory.assert_called_once_with(
            "s3", endpoint_url="http://localhost:9000", region_name=expected_region
        )
        client.head_bucket.assert_called_once_with(Bucket="glaze-test")

    def test_default_region_uses_auto_and_checks_bucket_readiness(self):
        self._assert_region_and_readiness(self._environment(), "auto")

    def test_explicit_region_is_used_and_checks_bucket_readiness(self):
        self._assert_region_and_readiness(
            self._environment(GLAZE_S3_REGION="us-east-1"), "us-east-1"
        )


if __name__ == "__main__":
    unittest.main()
