"""Local, pixel-bounded Glaze-style image protection inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Callable


MODEL_REPO = "STHCD/stable-diffusion-2-1-base"
MODEL_REVISION = "c3d96f8efa45b52299a2747849f6d95fb42219f6"
# Bump this when model use, preprocessing, loss, projection, output encoding, or request lifecycle changes.
ENGINE_VERSION = "glaze-targeted-pgd-v5"
MAX_IMAGE_SIDE = 1024
MIN_LPIPS_SIDE = 67
PADDING_RGB = (127, 127, 127)

ENGINE_SETTINGS = {
    "version": ENGINE_VERSION,
    "model_repo": MODEL_REPO,
    "model_revision": MODEL_REVISION,
    "optimizer": "Adam",
    "latent_loss": "mean_squared_vae_posterior_mean",
    "lpips_network": "alex",
    "maximum_input_side": MAX_IMAGE_SIDE,
    "minimum_lpips_side": MIN_LPIPS_SIDE,
    "large_mps_memory_policy": "MPS padded max side >512: auto attention slicing for target; offload UNet/text encoder/safety checker to CPU during optimization; restore before each target; synchronize and empty MPS cache",
    "mps_request_cache_cleanup": "empty unused MPS cache after every request, including failures and cancellation",
    "padding_rgb": PADDING_RGB,
    "padding_perturbation": 0.0,
    "output": "lossless_exact_webp",
}

__all__ = ["InvalidImageError", "InvalidSettingsError", "ProtectionEngine", "ProtectionSettings"]


class InvalidImageError(ValueError):
    """Raised when a client image cannot be processed under the image contract."""


class InvalidSettingsError(ValueError):
    """Raised when request protection settings do not satisfy the settings contract."""


@dataclass(frozen=True)
class ProtectionSettings:
    max_delta: int = 6
    iterations: int = 500
    learning_rate: float = 0.01
    lpips_threshold: float = 0.05
    lpips_weight: float = 10.0
    target_prompt: str = "oil painting by Van Gogh"
    target_seed: int = 12345
    target_steps: int = 30
    target_strength: float = 0.8
    target_guidance: float = 7.5

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ProtectionSettings:
        if not isinstance(value, Mapping):
            raise InvalidSettingsError("settings must be a JSON object.")

        allowed = {
            "max_delta", "iterations", "learning_rate", "lpips_threshold", "lpips_weight",
            "target_prompt", "target_seed", "target_steps", "target_strength", "target_guidance",
        }
        unknown = [key for key in value if key not in allowed]
        if unknown:
            raise InvalidSettingsError(f"Unknown protection settings: {', '.join(map(str, unknown))}.")

        def integer(name: str, default: int, minimum: int, maximum: int | None = None) -> int:
            result = value.get(name, default)
            if type(result) is not int or result < minimum or (maximum is not None and result > maximum):
                bounds = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
                raise InvalidSettingsError(f"{name} must be an integer in {bounds}.")
            return result

        def number(name: str, default: float, *, minimum: float, include_minimum: bool) -> float:
            result = value.get(name, default)
            if isinstance(result, bool) or not isinstance(result, (int, float)):
                raise InvalidSettingsError(f"{name} must be a finite number.")
            try:
                normalized = float(result)
            except (OverflowError, ValueError) as exc:
                raise InvalidSettingsError(f"{name} must be a finite number.") from exc
            if not math.isfinite(normalized):
                raise InvalidSettingsError(f"{name} must be a finite number.")
            if normalized < minimum or (not include_minimum and normalized == minimum):
                operator = ">=" if include_minimum else ">"
                raise InvalidSettingsError(f"{name} must be {operator} {minimum}.")
            return normalized

        max_delta = integer("max_delta", 6, 0, 255)
        iterations = integer("iterations", 500, 1)
        learning_rate = number("learning_rate", 0.01, minimum=0.0, include_minimum=False)
        lpips_threshold = number("lpips_threshold", 0.05, minimum=0.0, include_minimum=True)
        lpips_weight = number("lpips_weight", 10.0, minimum=0.0, include_minimum=True)
        target_prompt = value.get("target_prompt", "oil painting by Van Gogh")
        if not isinstance(target_prompt, str) or not target_prompt.strip():
            raise InvalidSettingsError("target_prompt must be a nonempty string.")
        target_seed = integer("target_seed", 12345, 0, 2**63 - 1)
        target_steps = integer("target_steps", 30, 1)
        target_strength = number("target_strength", 0.8, minimum=0.0, include_minimum=False)
        if target_strength > 1.0:
            raise InvalidSettingsError("target_strength must be <= 1.0.")
        try:
            has_effective_step = int(target_steps * target_strength) >= 1
        except OverflowError:
            numerator, denominator = target_strength.as_integer_ratio()
            has_effective_step = target_steps * numerator >= denominator
        if not has_effective_step:
            raise InvalidSettingsError("target_steps * target_strength must produce at least one inference step.")
        target_guidance = number("target_guidance", 7.5, minimum=0.0, include_minimum=True)

        return cls(
            max_delta=max_delta,
            iterations=iterations,
            learning_rate=learning_rate,
            lpips_threshold=lpips_threshold,
            lpips_weight=lpips_weight,
            target_prompt=target_prompt,
            target_seed=target_seed,
            target_steps=target_steps,
            target_strength=target_strength,
            target_guidance=target_guidance,
        )

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "max_delta": self.max_delta,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "lpips_threshold": self.lpips_threshold,
            "lpips_weight": self.lpips_weight,
            "target_prompt": self.target_prompt,
            "target_seed": self.target_seed,
            "target_steps": self.target_steps,
            "target_strength": self.target_strength,
            "target_guidance": self.target_guidance,
        }


def _validate_size(size: tuple[int, int]) -> None:
    width, height = size
    if width < 1 or height < 1:
        raise InvalidImageError("Image dimensions must be positive.")
    if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
        raise InvalidImageError(f"Image sides must be at most {MAX_IMAGE_SIDE} pixels.")


def _decode_image(image_bytes: bytes) -> tuple[Any, bool]:
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not isinstance(image_bytes, bytes):
        raise InvalidImageError("image_bytes must be bytes.")
    if not image_bytes:
        raise InvalidImageError("Image data is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = image.format
            if image_format not in {"JPEG", "PNG", "WEBP"}:
                raise InvalidImageError("Only still JPEG, PNG, and WebP images are supported.")
            if getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) > 1:
                raise InvalidImageError("Animated images are not supported.")
            _validate_size(image.size)
            image.verify()

        with Image.open(BytesIO(image_bytes)) as image:
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            oriented = ImageOps.exif_transpose(image)
            _validate_size(oriented.size)
            converted = oriented.convert("RGBA" if has_alpha else "RGB")
            converted.load()
            return converted.copy(), has_alpha
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, EOFError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("Image data is corrupt or unsupported.") from exc


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _prepare_canvas(
    image: Any,
    vae_scale_factor: int,
) -> tuple[Any, Any, tuple[int, int]]:
    import numpy as np
    from PIL import Image

    if vae_scale_factor < 1:
        raise ValueError("VAE scale factor must be positive.")
    width, height = image.size
    canvas_width = _round_up(max(width, MIN_LPIPS_SIDE), vae_scale_factor)
    canvas_height = _round_up(max(height, MIN_LPIPS_SIDE), vae_scale_factor)
    left = (canvas_width - width) // 2
    top = (canvas_height - height) // 2

    canvas = Image.new("RGB", (canvas_width, canvas_height), PADDING_RGB)
    canvas.paste(image.convert("RGB"), (left, top))
    padding_mask = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    padding_mask[top:top + height, left:left + width] = 1.0
    return canvas, padding_mask, (left, top)


def _encode_webp_checked(original: Any, protected_rgb: Any, max_delta: int) -> bytes:
    import numpy as np
    from PIL import Image, features

    if not features.check("webp"):
        raise RuntimeError("This Pillow build does not support WebP encoding.")
    if original.size != protected_rgb.size:
        raise ValueError("Protected pixels must have the original image dimensions.")

    has_alpha = original.mode == "RGBA"
    if protected_rgb.mode != "RGB":
        protected_rgb = protected_rgb.convert("RGB")
    if has_alpha:
        output_image = Image.merge("RGBA", (*protected_rgb.split(), original.getchannel("A")))
    else:
        output_image = protected_rgb

    buffer = BytesIO()
    output_image.save(
        buffer,
        format="WEBP",
        lossless=True,
        quality=100,
        method=6,
        exact=True,
    )
    encoded = buffer.getvalue()

    with Image.open(BytesIO(encoded)) as round_trip:
        round_trip.load()
        if round_trip.format != "WEBP" or round_trip.size != original.size:
            raise RuntimeError("Lossless WebP round-trip changed the image dimensions or format.")
        if has_alpha and "A" not in round_trip.getbands():
            raise RuntimeError("Lossless WebP round-trip dropped the alpha channel.")
        if not has_alpha and "A" in round_trip.getbands():
            raise RuntimeError("Lossless WebP round-trip changed the image alpha semantics.")

        original_rgb = np.asarray(original.convert("RGB"), dtype=np.int16)
        saved_rgb = np.asarray(round_trip.convert("RGB"), dtype=np.int16)
        expected_rgb = np.asarray(protected_rgb, dtype=np.uint8)
        if not np.array_equal(np.asarray(round_trip.convert("RGB"), dtype=np.uint8), expected_rgb):
            raise RuntimeError("Lossless WebP round-trip changed RGB values.")
        observed_delta = int(np.abs(saved_rgb - original_rgb).max())
        if observed_delta > max_delta:
            raise RuntimeError(
                f"Lossless WebP round-trip exceeded the {max_delta}-level RGB bound ({observed_delta})."
            )
        if has_alpha:
            original_alpha = np.asarray(original.getchannel("A"), dtype=np.uint8)
            saved_alpha = np.asarray(round_trip.getchannel("A"), dtype=np.uint8)
            if not np.array_equal(saved_alpha, original_alpha):
                raise RuntimeError("Lossless WebP round-trip changed alpha values.")

    return encoded


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_sha256(model_dir: Path) -> str:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local model snapshot directory does not exist: {model_dir}")

    files = sorted(
        (path for path in model_dir.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(model_dir).as_posix(),
    )
    if not files:
        raise ValueError("Local model snapshot directory is empty.")

    digest = sha256()
    for path in files:
        relative_name = path.relative_to(model_dir).as_posix().encode("utf-8")
        digest.update(len(relative_name).to_bytes(8, "big"))
        digest.update(relative_name)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _state_dict_sha256(state_dict: dict[str, object]) -> str:
    import torch

    digest = sha256()
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        metadata = f"{name}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode("utf-8")
        digest.update(metadata)
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_local_lpips(alexnet_weights: Path, device: str):
    import lpips
    import torch
    from torch import nn
    from torchvision.models import alexnet

    if not alexnet_weights.is_file():
        raise FileNotFoundError(f"Pre-provisioned AlexNet weights not found: {alexnet_weights}")

    weights = torch.load(alexnet_weights, map_location="cpu", weights_only=True)
    alexnet_model = alexnet(weights=None)
    alexnet_model.load_state_dict(weights, strict=True)

    # LPIPS ships its AlexNet calibration locally. Keep its trunk random until
    # the explicitly provisioned torchvision ImageNet weights are installed.
    model = lpips.LPIPS(net="alex", pnet_rand=True, verbose=False).eval()
    features_by_index = list(alexnet_model.features.children())
    for name, start, stop in (
        ("slice1", 0, 2),
        ("slice2", 2, 5),
        ("slice3", 5, 8),
        ("slice4", 8, 10),
        ("slice5", 10, 12),
    ):
        setattr(model.net, name, nn.Sequential(*features_by_index[start:stop]))

    model.requires_grad_(False)
    return model.eval().to(device)


def _optimize_canvas(
    vae,
    lpips_model,
    source01,
    target01,
    padding_mask,
    settings: ProtectionSettings,
    progress: Callable[[int, int], None] | None,
) -> Any:
    import numpy as np
    import torch
    import torch.nn.functional as functional

    source = source01 * 2.0 - 1.0
    target = target01 * 2.0 - 1.0
    with torch.no_grad():
        target_mean = vae.encode(target).latent_dist.mean.detach()

    delta = torch.zeros_like(source01, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=settings.learning_rate)
    if progress is not None:
        progress(0, settings.iterations)

    for step in range(1, settings.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        adversarial01 = (source01 + delta).clamp(0.0, 1.0)
        adversarial = adversarial01 * 2.0 - 1.0
        adversarial_mean = vae.encode(adversarial).latent_dist.mean
        latent_loss = torch.mean((adversarial_mean - target_mean) ** 2)
        lpips_value = lpips_model(source, adversarial).mean()
        loss = latent_loss + settings.lpips_weight * functional.relu(lpips_value - settings.lpips_threshold)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-settings.max_delta / 255.0, settings.max_delta / 255.0)
            delta.mul_(padding_mask)

        if progress is not None:
            progress(step, settings.iterations)

    with torch.no_grad():
        quantized = torch.round((source01 + delta).clamp(0.0, 1.0) * 255.0)
        protected = quantized[0].permute(1, 2, 0).to(torch.int16).cpu().numpy()
    original = torch.round(source01[0].permute(1, 2, 0) * 255.0).to(torch.int16).cpu().numpy()
    lower = np.maximum(original - settings.max_delta, 0)
    upper = np.minimum(original + settings.max_delta, 255)
    return np.minimum(np.maximum(protected, lower), upper).astype(np.uint8)


class ProtectionEngine:
    """Loads local Stable Diffusion 2.1 and runs bounded per-request protection."""

    def __init__(self, model_dir: str, device: str, alexnet_weights: str):
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of 'cpu', 'cuda', or 'mps'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise RuntimeError("MPS was requested but is not available.")

        model_path = Path(model_dir)
        alexnet_path = Path(alexnet_weights)
        if not alexnet_path.is_file():
            raise FileNotFoundError(f"Pre-provisioned AlexNet weights not found: {alexnet_path}")

        self.device = device
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float32,
            local_files_only=True,
        ).to(device)
        self._generation_modules = [
            module
            for module in (
                getattr(self.pipeline, "unet", None),
                getattr(self.pipeline, "text_encoder", None),
                getattr(self.pipeline, "safety_checker", None),
            )
            if module is not None
        ]
        self._generation_modules_offloaded = False
        self.vae = self.pipeline.vae.eval()
        self.vae.requires_grad_(False)
        self.vae_scale_factor = int(getattr(self.pipeline, "vae_scale_factor", 8))
        self.lpips = _load_local_lpips(alexnet_path, device)

        settings = json.dumps(
            {**ENGINE_SETTINGS, "device": self.device, "vae_scale_factor": self.vae_scale_factor},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = sha256()
        digest.update(bytes.fromhex(_snapshot_sha256(model_path)))
        digest.update(bytes.fromhex(_sha256_file(alexnet_path)))
        digest.update(bytes.fromhex(_state_dict_sha256(self.lpips.state_dict())))
        digest.update(settings)
        self.identity = digest.hexdigest()

    def _restore_generation_modules(self) -> None:
        if not getattr(self, "_generation_modules_offloaded", False):
            return
        for module in getattr(self, "_generation_modules", ()):
            module.to(self.device)
        self._generation_modules_offloaded = False

    def _offload_generation_modules_to_cpu(self) -> None:
        modules = getattr(self, "_generation_modules", ())
        if not modules:
            return
        self._generation_modules_offloaded = True
        for module in modules:
            module.to("cpu")
        import torch

        torch.mps.synchronize()
        torch.mps.empty_cache()

    def protect(
        self,
        image_bytes: bytes,
        settings: ProtectionSettings,
        progress: Callable[[int, int], None] | None = None,
    ) -> bytes:
        import torch
        from PIL import Image

        processing_failed = False
        try:
            original, _has_alpha = _decode_image(image_bytes)
            canvas, mask, offset = _prepare_canvas(original, self.vae_scale_factor)
            width, height = original.size
            left, top = offset

            generator = torch.Generator(device="cpu").manual_seed(settings.target_seed)
            use_attention_slicing = self.device == "mps" and max(canvas.size) > 512
            self._restore_generation_modules()
            try:
                if use_attention_slicing:
                    self.pipeline.enable_attention_slicing("auto")
                with torch.inference_mode():
                    target = self.pipeline(
                        prompt=settings.target_prompt,
                        image=canvas,
                        num_inference_steps=settings.target_steps,
                        strength=settings.target_strength,
                        guidance_scale=settings.target_guidance,
                        generator=generator,
                    ).images[0].convert("RGB")
            finally:
                if use_attention_slicing:
                    self.pipeline.disable_attention_slicing()
            if target.size != canvas.size:
                raise RuntimeError("Img2img pipeline changed the padded image dimensions.")
            if use_attention_slicing:
                self._offload_generation_modules_to_cpu()

            source01 = _torch_from_pil(canvas, self.device)
            target01 = _torch_from_pil(target, self.device)
            padding_mask = torch.from_numpy(mask).view(1, 1, *mask.shape).to(self.device)
            protected_canvas = _optimize_canvas(
                self.vae,
                self.lpips,
                source01,
                target01,
                padding_mask,
                settings,
                progress,
            )
            crop = protected_canvas[top:top + height, left:left + width]
            protected_rgb = Image.fromarray(crop)
            return _encode_webp_checked(original, protected_rgb, settings.max_delta)
        except BaseException:
            processing_failed = True
            raise
        finally:
            if self.device == "mps":
                try:
                    torch.mps.empty_cache()
                except BaseException:
                    if not processing_failed:
                        raise


def _torch_from_pil(image: Any, device: str):
    import numpy as np
    import torch
    from PIL import Image

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
