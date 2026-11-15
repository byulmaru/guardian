"""Offline checks for image boundaries, output fidelity, and progress."""

from io import BytesIO
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, features

from glaze_worker.inference import (
    InvalidImageError,
    InvalidSettingsError,
    _decode_image,
    _encode_webp_checked,
    _optimize_canvas,
    _prepare_canvas,
    ProtectionEngine,
    ProtectionSettings,
    _snapshot_sha256,
)


def encode_image(image: Image.Image, format_name: str, **options) -> bytes:
    output = BytesIO()
    image.save(output, format=format_name, **options)
    return output.getvalue()


class FakeModule:
    def __init__(self, name: str, device: str, events: list[tuple]):
        self.name = name
        self.device = device
        self.events = events

    def to(self, device: str):
        self.device = device
        self.events.append(("move", self.name, device))
        return self


class RecordingPipeline:
    def __init__(self, device: str, events: list[tuple]):
        self.events = events
        self.unet = FakeModule("unet", device, events)
        self.text_encoder = FakeModule("text_encoder", device, events)
        self.safety_checker = FakeModule("safety_checker", device, events)
        self.vae = FakeModule("vae", device, events)
        self.options = None

    def enable_attention_slicing(self, mode: str):
        self.events.append(("enable_slicing", mode))

    def disable_attention_slicing(self):
        self.events.append(("disable_slicing",))

    def __call__(self, **options):
        self.options = options
        devices = (self.unet.device, self.text_encoder.device, self.safety_checker.device)
        self.events.append(("target", devices))
        return SimpleNamespace(images=[options["image"].copy()])


def make_fake_engine(device: str, events: list[tuple]):
    pipeline = RecordingPipeline(device, events)
    engine = object.__new__(ProtectionEngine)
    engine.device = device
    engine.pipeline = pipeline
    engine._generation_modules = [pipeline.unet, pipeline.text_encoder, pipeline.safety_checker]
    engine._generation_modules_offloaded = False
    engine.vae = pipeline.vae
    engine.lpips = object()
    engine.vae_scale_factor = 8
    return engine, pipeline


def protect_with_fake_runtime(engine, pipeline, size, events, optimize, settings=None):
    import torch

    image_bytes = encode_image(Image.new("RGB", size, (20, 40, 60)), "PNG")
    with patch("glaze_worker.inference._torch_from_pil", return_value=None), \
            patch("glaze_worker.inference._optimize_canvas", side_effect=optimize), \
            patch.object(torch.Tensor, "to", lambda tensor, *_args, **_kwargs: tensor), \
            patch("torch.mps.synchronize", side_effect=lambda: events.append(("sync",))), \
            patch("torch.mps.empty_cache", side_effect=lambda: events.append(("empty_cache",))):
        return engine.protect(image_bytes, settings or ProtectionSettings.from_dict({}))


class GlazeInferenceImageTests(unittest.TestCase):
    def test_protection_settings_defaults_and_normalized_types(self) -> None:
        settings = ProtectionSettings.from_dict({})
        self.assertEqual(
            settings.as_dict(),
            {
                "max_delta": 6,
                "iterations": 500,
                "learning_rate": 0.01,
                "lpips_threshold": 0.05,
                "lpips_weight": 10.0,
                "target_prompt": "oil painting by Van Gogh",
                "target_seed": 12345,
                "target_steps": 30,
                "target_strength": 0.8,
                "target_guidance": 7.5,
            },
        )
        self.assertIsInstance(settings.learning_rate, float)
        self.assertIsInstance(settings.target_guidance, float)

    def test_protection_settings_reject_invalid_fields(self) -> None:
        invalid_settings = (
            None,
            [],
            {"unknown": 1},
            {"max_delta": -1},
            {"max_delta": 256},
            {"max_delta": True},
            {"iterations": 0},
            {"iterations": False},
            {"learning_rate": 0},
            {"learning_rate": float("nan")},
            {"learning_rate": float("inf")},
            {"lpips_threshold": -0.1},
            {"lpips_threshold": float("nan")},
            {"lpips_weight": -1},
            {"lpips_weight": float("inf")},
            {"target_prompt": ""},
            {"target_prompt": "   "},
            {"target_prompt": 123},
            {"target_seed": -1},
            {"target_seed": 2**63},
            {"target_seed": True},
            {"target_steps": 0},
            {"target_strength": 0},
            {"target_strength": 1.01},
            {"target_strength": float("nan")},
            {"target_steps": 1, "target_strength": 0.8},
            {"target_guidance": -0.1},
            {"target_guidance": float("inf")},
        )
        for value in invalid_settings:
            with self.subTest(value=value), self.assertRaises(InvalidSettingsError):
                ProtectionSettings.from_dict(value)

    def test_protection_settings_accept_bounds_and_zero_weight(self) -> None:
        settings = ProtectionSettings.from_dict(
            {
                "max_delta": 0,
                "lpips_threshold": 0,
                "lpips_weight": 0,
                "target_seed": 2**63 - 1,
                "target_steps": 1,
                "target_strength": 1.0,
                "target_guidance": 0,
            }
        )
        self.assertEqual(settings.max_delta, 0)
        self.assertEqual(settings.target_seed, 2**63 - 1)
        self.assertEqual(settings.target_guidance, 0.0)

    def test_accepts_1024_and_rejects_1025_pixel_sides(self) -> None:
        accepted, _ = _decode_image(encode_image(Image.new("RGB", (1024, 1)), "PNG"))
        self.assertEqual(accepted.size, (1024, 1))

        for size in ((1025, 1), (1, 1025)):
            with self.subTest(size=size), self.assertRaises(InvalidImageError):
                _decode_image(encode_image(Image.new("RGB", size), "PNG"))

    def test_accepts_still_jpeg_png_and_webp(self) -> None:
        image = Image.new("RGB", (3, 2), (20, 40, 60))
        for format_name in ("JPEG", "PNG", "WEBP"):
            with self.subTest(format_name=format_name):
                if format_name == "WEBP" and not features.check("webp"):
                    self.skipTest("Pillow was built without WebP support.")
                decoded, has_alpha = _decode_image(encode_image(image, format_name))
                self.assertEqual(decoded.size, image.size)
                self.assertFalse(has_alpha)

    def test_preserves_exif_oriented_dimensions(self) -> None:
        image = Image.new("RGB", (4, 2), (20, 40, 60))
        exif = image.getexif()
        exif[274] = 6
        decoded, _ = _decode_image(encode_image(image, "JPEG", exif=exif))
        self.assertEqual(decoded.size, (2, 4))

    def test_decode_preserves_transparency_and_hidden_rgb(self) -> None:
        pixels = np.array([[[1, 2, 3, 0], [4, 5, 6, 127]]], dtype=np.uint8)
        decoded, has_alpha = _decode_image(encode_image(Image.fromarray(pixels), "PNG"))
        self.assertTrue(has_alpha)
        np.testing.assert_array_equal(np.asarray(decoded), pixels)

    def test_rejects_animated_and_corrupt_images(self) -> None:
        animated = encode_image(
            Image.new("RGB", (2, 2), "red"),
            "PNG",
            save_all=True,
            append_images=[Image.new("RGB", (2, 2), "blue")],
            duration=20,
            loop=0,
        )
        with self.assertRaisesRegex(InvalidImageError, "Animated"):
            _decode_image(animated)
        with self.assertRaises(InvalidImageError):
            _decode_image(b"not an image")

    def test_rejects_unsupported_and_mistyped_input_with_specific_error(self) -> None:
        self.assertTrue(issubclass(InvalidImageError, ValueError))
        with self.assertRaises(InvalidImageError):
            _decode_image(encode_image(Image.new("RGB", (1, 1)), "GIF"))
        with self.assertRaises(InvalidImageError):
            _decode_image(bytearray(b"not bytes"))

    def test_webp_round_trip_preserves_rgb_and_alpha(self) -> None:
        if not features.check("webp"):
            self.skipTest("Pillow was built without WebP support.")
        pixels = np.array(
            [
                [[100, 120, 140, 0], [10, 20, 30, 255]],
                [[250, 1, 2, 80], [60, 70, 80, 200]],
            ],
            dtype=np.uint8,
        )
        original = Image.fromarray(pixels)
        protected = np.clip(pixels[:, :, :3].astype(np.int16) + 6, 0, 255).astype(np.uint8)
        encoded = _encode_webp_checked(original, Image.fromarray(protected), max_delta=6)

        with Image.open(BytesIO(encoded)) as decoded:
            decoded.load()
            self.assertEqual(decoded.format, "WEBP")
            self.assertEqual(decoded.size, original.size)
            self.assertEqual(decoded.mode, "RGBA")
            np.testing.assert_array_equal(np.asarray(decoded), np.dstack((protected, pixels[:, :, 3])))

    def test_rgb_webp_round_trip_keeps_the_protected_pixels(self) -> None:
        if not features.check("webp"):
            self.skipTest("Pillow was built without WebP support.")
        original = Image.new("RGB", (2, 1), (10, 20, 30))
        protected = Image.new("RGB", (2, 1), (16, 14, 36))
        encoded = _encode_webp_checked(original, protected, max_delta=6)
        with Image.open(BytesIO(encoded)) as decoded:
            self.assertEqual(decoded.mode, "RGB")
            np.testing.assert_array_equal(np.asarray(decoded), np.asarray(protected))

    def test_webp_round_trip_rejects_channel_change_over_six(self) -> None:
        if not features.check("webp"):
            self.skipTest("Pillow was built without WebP support.")
        original = Image.new("RGB", (1, 1), (10, 20, 30))
        changed = Image.new("RGB", (1, 1), (17, 20, 30))
        with self.assertRaisesRegex(RuntimeError, "6-level"):
            _encode_webp_checked(original, changed, max_delta=6)

    def test_webp_round_trip_uses_request_delta_bound(self) -> None:
        if not features.check("webp"):
            self.skipTest("Pillow was built without WebP support.")
        original = Image.new("RGB", (1, 1), (10, 20, 30))
        protected = Image.new("RGB", (1, 1), (13, 20, 30))
        with self.assertRaisesRegex(RuntimeError, "2-level"):
            _encode_webp_checked(original, protected, max_delta=2)

    def test_small_canvas_pads_without_scaling_and_masks_padding(self) -> None:
        original = Image.new("RGB", (3, 2), (11, 22, 33))
        canvas, mask, (left, top) = _prepare_canvas(original, vae_scale_factor=8)
        self.assertEqual(canvas.size, (72, 72))
        self.assertEqual(canvas.getpixel((left, top)), (11, 22, 33))
        self.assertEqual(float(mask.sum()), 6.0)
        self.assertEqual(mask[top, left], 1.0)
        self.assertEqual(mask[0, 0], 0.0)

    def test_snapshot_identity_uses_contents_and_relative_names(self) -> None:
        with TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir:
            first = Path(first_dir)
            second = Path(second_dir)
            (first / "unet").mkdir()
            (second / "unet").mkdir()
            (first / "unet" / "weights.bin").write_bytes(b"same")
            (second / "unet" / "weights.bin").write_bytes(b"same")
            self.assertEqual(_snapshot_sha256(first), _snapshot_sha256(second))
            (second / "unet" / "weights.bin").write_bytes(b"different")
            self.assertNotEqual(_snapshot_sha256(first), _snapshot_sha256(second))

    def test_optimizer_reports_start_and_completed_steps(self) -> None:
        import torch

        class FakeVAE:
            def encode(self, image):
                mean = image.mean(dim=(2, 3), keepdim=True)
                return SimpleNamespace(latent_dist=SimpleNamespace(mean=mean))

        class FakeLPIPS:
            def __call__(self, original, protected):
                return (original - protected).square().mean(dim=(1, 2, 3))

        source = torch.full((1, 3, 8, 8), 0.4)
        target = torch.full((1, 3, 8, 8), 0.7)
        mask = torch.ones((1, 1, 8, 8))
        mask[:, :, :, -1] = 0.0
        progress = []

        protected = _optimize_canvas(
            FakeVAE(),
            FakeLPIPS(),
            source,
            target,
            mask,
            ProtectionSettings.from_dict({"iterations": 2}),
            lambda done, total: progress.append((done, total)),
        )

        self.assertEqual(progress, [(0, 2), (1, 2), (2, 2)])
        self.assertEqual(protected.shape, (8, 8, 3))
        self.assertTrue(np.all(protected[:, -1] == 102))
        self.assertLessEqual(int(np.abs(protected.astype(np.int16) - 102).max()), 6)

    def test_optimizer_applies_nondefault_learning_rate_lpips_and_delta(self) -> None:
        import torch
        import torch.nn.functional as functional

        class FakeVAE:
            def encode(self, image):
                mean = image.mean(dim=(2, 3), keepdim=True)
                return SimpleNamespace(latent_dist=SimpleNamespace(mean=mean))

        class FakeLPIPS:
            def __call__(self, _original, protected):
                return torch.full((1,), 0.2, device=protected.device)

        settings = ProtectionSettings.from_dict(
            {
                "max_delta": 2,
                "iterations": 1,
                "learning_rate": 0.02,
                "lpips_threshold": 0.1,
                "lpips_weight": 2.5,
            }
        )
        source = torch.full((1, 3, 8, 8), 0.4)
        target = torch.full((1, 3, 8, 8), 0.7)
        mask = torch.ones((1, 1, 8, 8))
        progress = []
        optimizer_learning_rates = []
        relu_inputs = []
        loss_values = []
        original_adam = torch.optim.Adam
        original_relu = functional.relu
        original_backward = torch.Tensor.backward

        def recording_adam(*args, **kwargs):
            optimizer_learning_rates.append(kwargs["lr"])
            return original_adam(*args, **kwargs)

        def recording_relu(value, *args, **kwargs):
            relu_inputs.append(float(value.detach()))
            return original_relu(value, *args, **kwargs)

        def recording_backward(value, *args, **kwargs):
            loss_values.append(float(value.detach()))
            return original_backward(value, *args, **kwargs)

        with patch("torch.optim.Adam", side_effect=recording_adam), \
                patch("torch.nn.functional.relu", side_effect=recording_relu), \
                patch.object(torch.Tensor, "backward", recording_backward):
            protected = _optimize_canvas(
                FakeVAE(),
                FakeLPIPS(),
                source,
                target,
                mask,
                settings,
                lambda done, total: progress.append((done, total)),
            )

        self.assertEqual(optimizer_learning_rates, [0.02])
        self.assertAlmostEqual(relu_inputs[0], 0.1, places=6)
        self.assertAlmostEqual(loss_values[0], 0.36 + 2.5 * 0.1, places=5)
        self.assertEqual(progress, [(0, 1), (1, 1)])
        self.assertLessEqual(int(np.abs(protected.astype(np.int16) - 102).max()), 2)

    def test_protect_orchestrates_fake_pipeline_through_lossless_output(self) -> None:
        class FakePipeline:
            def __init__(self):
                self.options = None

            def __call__(self, **options):
                self.options = options
                return SimpleNamespace(images=[options["image"].copy()])

        source_pixels = np.array(
            [
                [[10, 20, 30, 0], [50, 60, 70, 128], [80, 90, 100, 255]],
                [[110, 120, 130, 255], [140, 150, 160, 64], [170, 180, 190, 0]],
            ],
            dtype=np.uint8,
        )
        source_bytes = encode_image(Image.fromarray(source_pixels), "PNG")
        fake_pipeline = FakePipeline()
        engine = object.__new__(ProtectionEngine)
        engine.device = "cpu"
        engine.pipeline = fake_pipeline
        engine.vae = object()
        engine.lpips = object()
        engine.vae_scale_factor = 8
        progress = []

        settings = ProtectionSettings.from_dict(
            {
                "max_delta": 2,
                "iterations": 3,
                "target_prompt": "a watercolor of mountains",
                "target_seed": 77,
                "target_steps": 5,
                "target_strength": 0.6,
                "target_guidance": 4.25,
            }
        )

        def fake_optimize(vae, lpips_model, source01, target01, padding_mask, actual_settings, callback):
            self.assertIs(vae, engine.vae)
            self.assertIs(lpips_model, engine.lpips)
            self.assertEqual(actual_settings, settings)
            self.assertEqual(tuple(source01.shape), (1, 3, 72, 72))
            self.assertEqual(tuple(target01.shape), (1, 3, 72, 72))
            self.assertEqual(tuple(padding_mask.shape), (1, 1, 72, 72))
            canvas = np.asarray(fake_pipeline.options["image"], dtype=np.uint8)
            protected = canvas.copy()
            left, top = 34, 35
            protected[top:top + 2, left:left + 3] += 2
            callback(0, actual_settings.iterations)
            callback(actual_settings.iterations, actual_settings.iterations)
            return protected

        with patch("glaze_worker.inference._optimize_canvas", side_effect=fake_optimize):
            output = engine.protect(source_bytes, settings, lambda done, total: progress.append((done, total)))

        self.assertEqual(fake_pipeline.options["prompt"], "a watercolor of mountains")
        self.assertEqual(fake_pipeline.options["num_inference_steps"], 5)
        self.assertEqual(fake_pipeline.options["strength"], 0.6)
        self.assertEqual(fake_pipeline.options["guidance_scale"], 4.25)
        self.assertEqual(fake_pipeline.options["generator"].initial_seed(), 77)
        self.assertEqual(progress, [(0, 3), (3, 3)])
        with Image.open(BytesIO(output)) as saved:
            saved.load()
            self.assertEqual(saved.format, "WEBP")
            self.assertEqual(saved.size, (3, 2))
            self.assertEqual(saved.mode, "RGBA")
            expected = source_pixels.copy()
            expected[:, :, :3] += 2
            np.testing.assert_array_equal(np.asarray(saved), expected)

    def test_attention_slicing_is_limited_to_large_mps_target_generation(self) -> None:
        cases = (
            (
                "mps",
                (1024, 1024),
                [
                    ("enable_slicing", "auto"),
                    ("target", ("mps", "mps", "mps")),
                    ("disable_slicing",),
                    ("move", "unet", "cpu"),
                    ("move", "text_encoder", "cpu"),
                    ("move", "safety_checker", "cpu"),
                    ("sync",),
                    ("empty_cache",),
                    ("optimize", ("cpu", "cpu", "cpu")),
                    ("empty_cache",),
                ],
                ("cpu", "cpu", "cpu"),
            ),
            ("mps", (512, 512), [("target", ("mps", "mps", "mps")), ("optimize", ("mps", "mps", "mps")), ("empty_cache",)], ("mps", "mps", "mps")),
            ("cpu", (1024, 1024), [("target", ("cpu", "cpu", "cpu")), ("optimize", ("cpu", "cpu", "cpu"))], ("cpu", "cpu", "cpu")),
            ("cuda", (1024, 1024), [("target", ("cuda", "cuda", "cuda")), ("optimize", ("cuda", "cuda", "cuda"))], ("cuda", "cuda", "cuda")),
        )
        for device, size, expected_events, expected_module_devices in cases:
            with self.subTest(device=device, size=size):
                events = []
                engine, pipeline = make_fake_engine(device, events)

                def return_canvas(_vae, _lpips, _source, _target, _mask, _settings, _progress):
                    module_devices = (pipeline.unet.device, pipeline.text_encoder.device, pipeline.safety_checker.device)
                    events.append(("optimize", module_devices))
                    return np.asarray(pipeline.options["image"], dtype=np.uint8).copy()

                output = protect_with_fake_runtime(engine, pipeline, size, events, return_canvas)

                self.assertEqual(events, expected_events)
                self.assertEqual(
                    (pipeline.unet.device, pipeline.text_encoder.device, pipeline.safety_checker.device),
                    expected_module_devices,
                )
                self.assertEqual(pipeline.vae.device, device)
                with Image.open(BytesIO(output)) as saved:
                    self.assertEqual(saved.format, "WEBP")
                    self.assertEqual(saved.size, size)

    def test_large_mps_modules_restore_before_the_next_target(self) -> None:
        events = []
        engine, pipeline = make_fake_engine("mps", events)

        def return_canvas(_vae, _lpips, _source, _target, _mask, _settings, _progress):
            module_devices = (pipeline.unet.device, pipeline.text_encoder.device, pipeline.safety_checker.device)
            events.append(("optimize", module_devices))
            return np.asarray(pipeline.options["image"], dtype=np.uint8).copy()

        protect_with_fake_runtime(engine, pipeline, (1024, 1024), events, return_canvas)
        self.assertTrue(engine._generation_modules_offloaded)
        self.assertEqual(
            (pipeline.unet.device, pipeline.text_encoder.device, pipeline.safety_checker.device),
            ("cpu", "cpu", "cpu"),
        )

        events.clear()
        protect_with_fake_runtime(engine, pipeline, (512, 512), events, return_canvas)
        self.assertEqual(
            events,
            [
                ("move", "unet", "mps"),
                ("move", "text_encoder", "mps"),
                ("move", "safety_checker", "mps"),
                ("target", ("mps", "mps", "mps")),
                ("optimize", ("mps", "mps", "mps")),
                ("empty_cache",),
            ],
        )
        self.assertFalse(engine._generation_modules_offloaded)
        self.assertEqual(pipeline.vae.device, "mps")

    def test_mps_modules_restore_after_optimization_error_or_cancellation(self) -> None:
        for failure in (RuntimeError("optimization failed"), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                events = []
                engine, pipeline = make_fake_engine("mps", events)
                attempts = 0

                def optimize(_vae, _lpips, _source, _target, _mask, _settings, _progress):
                    nonlocal attempts
                    attempts += 1
                    devices = (pipeline.unet.device, pipeline.text_encoder.device, pipeline.safety_checker.device)
                    events.append(("optimize", devices))
                    if attempts == 1:
                        raise failure
                    return np.asarray(pipeline.options["image"], dtype=np.uint8).copy()

                with self.assertRaises(type(failure)):
                    protect_with_fake_runtime(engine, pipeline, (1024, 1024), events, optimize)
                self.assertTrue(engine._generation_modules_offloaded)
                self.assertEqual(events[:3], [("enable_slicing", "auto"), ("target", ("mps",) * 3), ("disable_slicing",)])
                self.assertEqual(events[-4:], [
                    ("sync",),
                    ("empty_cache",),
                    ("optimize", ("cpu",) * 3),
                    ("empty_cache",),
                ])

                events.clear()
                protect_with_fake_runtime(engine, pipeline, (512, 512), events, optimize)
                self.assertEqual(events[:4], [
                    ("move", "unet", "mps"),
                    ("move", "text_encoder", "mps"),
                    ("move", "safety_checker", "mps"),
                    ("target", ("mps",) * 3),
                ])
                self.assertEqual(events[-2:], [("optimize", ("mps",) * 3), ("empty_cache",)])
                self.assertFalse(engine._generation_modules_offloaded)

    def test_mps_attention_slicing_is_disabled_after_generation_failure_or_cancel(self) -> None:
        class FakePipeline:
            def __init__(self, failure):
                self.failure = failure
                self.events = []

            def enable_attention_slicing(self, mode):
                self.events.append(("enable", mode))

            def disable_attention_slicing(self):
                self.events.append(("disable",))

            def __call__(self, **_options):
                self.events.append(("target",))
                raise self.failure

        source_bytes = encode_image(Image.new("RGB", (1024, 1024), (20, 40, 60)), "PNG")
        for failure in (RuntimeError("generation failed"), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                pipeline = FakePipeline(failure)
                engine = object.__new__(ProtectionEngine)
                engine.device = "mps"
                engine.pipeline = pipeline
                engine.vae = object()
                engine.lpips = object()
                engine.vae_scale_factor = 8
                with patch("torch.mps.empty_cache", side_effect=lambda: pipeline.events.append(("empty_cache",))):
                    with self.assertRaises(type(failure)):
                        engine.protect(source_bytes, ProtectionSettings.from_dict({}))
                self.assertEqual(pipeline.events, [
                    ("enable", "auto"),
                    ("target",),
                    ("disable",),
                    ("empty_cache",),
                ])

    def test_mps_cache_is_cleared_for_invalid_input(self) -> None:
        events = []
        engine, _pipeline = make_fake_engine("mps", events)

        with patch("torch.mps.empty_cache", side_effect=lambda: events.append(("empty_cache",))):
            with self.assertRaises(InvalidImageError):
                engine.protect(b"not an image", ProtectionSettings.from_dict({}))

        self.assertEqual(events, [("empty_cache",)])

    def test_mps_cleanup_failure_does_not_mask_processing_failure_or_succeed_silently(self) -> None:
        cleanup_failure = RuntimeError("MPS cache cleanup failed")
        events = []
        engine, _pipeline = make_fake_engine("mps", events)

        with patch("torch.mps.empty_cache", side_effect=cleanup_failure):
            with self.assertRaisesRegex(InvalidImageError, "corrupt or unsupported"):
                engine.protect(b"not an image", ProtectionSettings.from_dict({}))

        engine, pipeline = make_fake_engine("mps", events)
        import torch

        def return_canvas(_vae, _lpips, _source, _target, _mask, _settings, _progress):
            return np.asarray(pipeline.options["image"], dtype=np.uint8).copy()

        image_bytes = encode_image(Image.new("RGB", (512, 512), (20, 40, 60)), "PNG")
        with patch("glaze_worker.inference._torch_from_pil", return_value=None), \
                patch("glaze_worker.inference._optimize_canvas", side_effect=return_canvas), \
                patch.object(torch.Tensor, "to", lambda tensor, *_args, **_kwargs: tensor), \
                patch("torch.mps.empty_cache", side_effect=cleanup_failure):
            with self.assertRaisesRegex(RuntimeError, "MPS cache cleanup failed"):
                engine.protect(image_bytes, ProtectionSettings.from_dict({}))

if __name__ == "__main__":
    unittest.main()
