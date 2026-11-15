# Glaze Temporal Activity worker

This standalone Python 3.12 worker runs the Glaze protection Activity. It accepts a JSON payload with `source_key`, `result_key`, and a required `settings` object, and returns `{"result_key": "..."}`. Every field inside `settings` is optional and uses the default shown below when omitted. The worker reads and writes objects in the configured S3 bucket; it does not train models.

The processing contract preserves input dimensions, rejects images larger than 1024 pixels on either side, and stores lossless WebP. Retries for the same source object version and effective settings match, so omitting a setting is equivalent to supplying its default. Reusing a `result_key` with different effective settings produces a non-retryable conflict; existing results are not overwritten.

The 1024-pixel limit is a processing bound, not a performance guarantee; runtime and throughput have not been profiled.

## Per-request style and optimizer settings

The `settings` object is required, but its keys are optional:

| Key | Accepted value | Default | Effect |
| --- | --- | --- | --- |
| `max_delta` | Integer from 0 through 255 | `6` | Maximum integer change allowed for each RGB channel relative to the source pixel. This channel-level bound does not guarantee a perceptual similarity or style outcome. |
| `iterations` | Positive integer | `500` | Number of optimizer update steps. |
| `learning_rate` | Positive finite number | `0.01` | Adam optimizer step size for each update. It controls update size; `iterations` controls how many updates run. |
| `lpips_threshold` | Number greater than or equal to 0 | `0.05` | LPIPS hinge threshold: the visual-similarity penalty applies to LPIPS values above this threshold. |
| `lpips_weight` | Number greater than or equal to 0 | `10` | Scales the visual-similarity penalty above `lpips_threshold`; it is distinct from the optimizer learning rate. |
| `target_prompt` | Non-empty string | `"oil painting by Van Gogh"` | Prompt used to generate the optimization target. |
| `target_seed` | Integer from 0 through 9,223,372,036,854,775,807 | `12345` | Seed for target generation. TypeScript callers should use JavaScript safe integers so JSON preserves the value exactly. |
| `target_steps` | Positive integer | `30` | Number of target-generation steps. |
| `target_strength` | Number greater than 0 and at most 1, with `int(target_steps * target_strength) >= 1` | `0.8` | Strength for target generation. The combined step-count constraint must hold for the selected `target_steps`. |
| `target_guidance` | Number greater than or equal to 0 | `7.5` | Guidance scale for target generation. |

`learning_rate` controls the size of each optimizer update, while `iterations` controls the number of updates. `lpips_threshold` sets where the LPIPS penalty starts, and `lpips_weight` controls that penalty's influence. These settings do not make a particular style or level of protection guaranteed: target prompts are inputs to the target-generation step, and arbitrary style transfer or robustness has not been validated. Treat any presets or non-default combinations as experimental; their effects are unknown.

## Model files and notices

Provision the model snapshot as a local directory before starting the worker. Set `GLAZE_MODEL_DIR` to that directory inside the process or container. Startup loads this local snapshot and does not download a model.

LPIPS also needs a separately provisioned torchvision AlexNet `IMAGENET1K_V1` `.pth` file. Set `GLAZE_ALEXNET_WEIGHTS` to its absolute path. The weight file is not bundled with the model snapshot or image, and startup does not download it.

Preserve the notices supplied with model and weight files. If a model snapshot was obtained from a mirror, that mirror is not an official model source. This project does not establish the mirror's provenance or license authority; review the applicable terms before using or distributing the model, weights, or an image containing them.

## Install and run locally

Use the official PyTorch CPU wheel index for the CPU environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple -e .
```

Set these values for the worker process:

| Variable | Value |
| --- | --- |
| `GLAZE_MODEL_DIR` | Absolute path to the pre-provisioned local model snapshot. |
| `GLAZE_ALEXNET_WEIGHTS` | Absolute path to the separate AlexNet `IMAGENET1K_V1` `.pth` file. |
| `GLAZE_DEVICE` | Inference device, such as `cpu`. |
| `GLAZE_S3_BUCKET` | Bucket containing source and result objects. |
| `GLAZE_S3_ENDPOINT` | Required HTTP(S) S3 API endpoint, such as `https://your-account-id.r2.cloudflarestorage.com` for R2 or `https://s3.us-east-1.amazonaws.com` for AWS S3. Local S3-compatible endpoints may use HTTP. |
| `GLAZE_S3_REGION` | Optional boto3 signing region; defaults to `auto` for the existing R2 behavior. Set it to the AWS bucket's region, such as `us-east-1`, when using AWS S3. |
| `TEMPORAL_ADDRESS` | Temporal frontend address, including port. |
| `TEMPORAL_NAMESPACE` | Temporal namespace. |
| `GLAZE_TASK_QUEUE` | Activity task queue; defaults to `glaze-protect`. |

Provide AWS credentials through the runtime AWS credential chain, such as the container's assigned role. Credentials are not build arguments or image contents.

## Call the Activity from TypeScript

This example invokes the Python Activity directly from a Temporal TypeScript Workflow. Configure the Workflow constant to match the worker's `GLAZE_TASK_QUEUE` setting; the worker defaults to `glaze-protect`.

```ts
import { proxyActivities } from '@temporalio/workflow';

type GlazeSettings = {
  max_delta?: number;
  iterations?: number;
  learning_rate?: number;
  lpips_threshold?: number;
  lpips_weight?: number;
  target_prompt?: string;
  target_seed?: number;
  target_steps?: number;
  target_strength?: number;
  target_guidance?: number;
};

type GlazeRequest = {
  source_key: string;
  result_key: string;
  settings: GlazeSettings;
};

type GlazeActivities = {
  protect_image(input: GlazeRequest): Promise<{ result_key: string }>;
};

const GLAZE_TASK_QUEUE = 'glaze-protect'; // Match the worker's GLAZE_TASK_QUEUE.

const { protect_image } = proxyActivities<GlazeActivities>({
  taskQueue: GLAZE_TASK_QUEUE,
  startToCloseTimeout: '60 minutes',
  heartbeatTimeout: '2 minutes',
});

export async function protectImageWorkflow(input: GlazeRequest): Promise<string> {
  const result = await protect_image(input);
  return result.result_key;
}
```

The timeout values are examples; profile the model and target hardware before relying on them. This illustrative snippet does not add a production Workflow or automatically select Media for protection. See Temporal's [`proxyActivities`](https://typescript.temporal.io/api/namespaces/workflow) and [`ActivityOptions`](https://typescript.temporal.io/api/interfaces/common.ActivityOptions/) references.

Start the worker with:

```sh
python -m glaze_worker.worker
```

## Build and run the container

The Dockerfile installs the CPU-only PyTorch wheel. The Docker build context includes `pyproject.toml` and `glaze_worker/`.

```sh
docker build --file Dockerfile.glaze-worker --tag glaze-worker:local .
```

Run with read-only mounts for both model inputs:

```sh
docker run --rm \
  --mount "type=bind,src=/absolute/path/to/model-snapshot,dst=/models/glaze,readonly" \
  --mount "type=bind,src=/absolute/path/to/alexnet-imagenet1k-v1.pth,dst=/weights/alexnet.pth,readonly" \
  -e GLAZE_MODEL_DIR=/models/glaze \
  -e GLAZE_ALEXNET_WEIGHTS=/weights/alexnet.pth \
  -e GLAZE_DEVICE=cpu \
  -e GLAZE_S3_BUCKET=your-bucket \
  -e GLAZE_S3_ENDPOINT=https://s3.us-east-1.amazonaws.com \
  -e GLAZE_S3_REGION=us-east-1 \
  -e TEMPORAL_ADDRESS=your-temporal-host:7233 \
  -e TEMPORAL_NAMESPACE=default \
  -e GLAZE_TASK_QUEUE=glaze-protect \
  glaze-worker:local
```

The container runs as a non-root user. It does not include credentials or model files. The image built from `Dockerfile.glaze-worker` is CPU-only; Linux containers do not expose Apple MPS. Use the separate CUDA image below for an NVIDIA GPU.

## Build and run the NVIDIA CUDA container

`Dockerfile.glaze-worker.cuda` builds a separate CUDA image for 64-bit Linux NVIDIA instances on both `amd64` and `arm64`. It keeps the existing PyTorch 2.6.0+cu124 and torchvision 0.21.0+cu124 pair on `amd64`; `arm64` uses PyTorch 2.6.0+cu126 and torchvision 0.21.0 from the official [CUDA 12.6 torch](https://download.pytorch.org/whl/cu126/torch/) and [torchvision](https://download.pytorch.org/whl/cu126/torchvision/) wheel indexes. Those indexes provide Python 3.12 Linux ARM64 wheels for the project's pinned PyTorch 2.6 and torchvision 0.21 releases. The version constraints prevent pip from replacing the CUDA-enabled PyTorch wheel with a CPU wheel. The original `Dockerfile.glaze-worker` remains CPU-only. The CUDA image does not claim support for AMD GPUs.

The host needs an NVIDIA driver compatible with the CUDA runtime for its architecture (12.4 on `amd64`, 12.6 on `arm64`) and the NVIDIA Container Toolkit configured for Docker. The image runs as UID 10001. Provision the model snapshot and AlexNet weights outside the image, ensure UID 10001 can read them, and mount both read-only. The CUDA image does not include model files, AWS credentials, or Temporal credentials. Inject credentials through the runtime's configured AWS and Temporal credential mechanisms; never pass them as build arguments or bake them into the image. Review this repository's code license and the applicable model and weight terms before using or redistributing the image.

Build for the target NVIDIA host architecture. The Dockerfile selects the corresponding official PyTorch CUDA wheel index:

```sh
docker build --platform linux/arm64 \
  --file Dockerfile.glaze-worker.cuda \
  --tag glaze-worker:cuda .
```

For an `amd64` host, use `--platform linux/amd64`; for an `arm64` host, use `--platform linux/arm64`. Run with GPU access and the same worker settings used by the CPU container:

```sh
docker run --rm --gpus all --platform linux/arm64 \
  --mount "type=bind,src=/absolute/path/to/model-snapshot,dst=/models/glaze,readonly" \
  --mount "type=bind,src=/absolute/path/to/alexnet-imagenet1k-v1.pth,dst=/weights/alexnet.pth,readonly" \
  -e GLAZE_MODEL_DIR=/models/glaze \
  -e GLAZE_ALEXNET_WEIGHTS=/weights/alexnet.pth \
  -e GLAZE_DEVICE=cuda \
  -e GLAZE_S3_BUCKET=your-bucket \
  -e GLAZE_S3_ENDPOINT=https://s3.us-east-1.amazonaws.com \
  -e GLAZE_S3_REGION=us-east-1 \
  -e TEMPORAL_ADDRESS=your-temporal-host:7233 \
  -e TEMPORAL_NAMESPACE=default \
  -e GLAZE_TASK_QUEUE=glaze-protect \
  glaze-worker:cuda
```

The CUDA build context includes `pyproject.toml` and `glaze_worker/`. GPU availability depends on the host driver, NVIDIA Container Toolkit, and instance configuration. The image build and CI smoke check do not verify GPU execution on NVIDIA hardware.

## Tests

Install the optional test dependency and run the activity, inference, and storage unit tests. They do not require a live Temporal server or object store.

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple -e ".[test]"
python -m pytest -q tests/test_glaze_activity.py tests/test_glaze_inference.py tests/test_glaze_storage.py
```

## GitHub Actions and GHCR

`.github/workflows/glaze-worker-image.yml` builds and smoke-checks both architectures on native GitHub-hosted runners: `ubuntu-latest` for `linux/amd64` and `ubuntu-24.04-arm` for `linux/arm64`. The smoke checks verify the CUDA-enabled PyTorch package and worker imports without verifying GPU execution. Pull-request jobs have read-only repository permissions. Default-branch jobs push each architecture tag after its smoke check passes, then publish the multi-platform `ghcr.io/<owner>/<repository>/glaze-worker:<sha>` and `:latest` tags after both architectures pass. The per-architecture tags use `<sha>-amd64` and `<sha>-arm64`. Use lowercase owner and repository names; only the default-branch publishing jobs need `packages: write`. Running the image still requires a compatible NVIDIA driver and the NVIDIA Container Toolkit configured for Docker.
