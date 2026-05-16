import os
from huggingface_hub import HfFileSystem
from transformers import TrainerCallback


HF_BUCKET = "hf://buckets/JimmyFu/FeatureDetectionAndDescriptionCheckpoint"


class HFBucketCallback(TrainerCallback):
    """
    Uploads checkpoints to a HuggingFace Storage Bucket after each save,
    keeping only two: checkpoint-latest (most recent save) and checkpoint-best
    (best metric so far). Provides a classmethod to download the latest
    checkpoint for resuming training on a fresh instance.
    Relies on HF_TOKEN environment variable or a prior `huggingface-cli login`.
    """

    def __init__(self, model_name, bucket=HF_BUCKET):
        self.bucket = bucket
        self.model_name = model_name
        self.bucket_path = f"{bucket}/{model_name}"
        self.fs = HfFileSystem()

    def _replace(self, local_dir, remote_path):
        if self.fs.exists(remote_path):
            self.fs.rm(remote_path, recursive=True)
        print(f"Uploading {local_dir} -> {remote_path}")
        self.fs.put(local_dir, remote_path, recursive=True)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(checkpoint_dir):
            return

        try:
            import wandb
            if wandb.run is not None:
                with open(os.path.join(checkpoint_dir, "wandb_run_id.txt"), "w") as f:
                    f.write(wandb.run.id)
        except ImportError:
            pass

        self._replace(checkpoint_dir, f"{self.bucket_path}/checkpoint-latest")

        if state.best_model_checkpoint and os.path.realpath(
            state.best_model_checkpoint
        ) == os.path.realpath(checkpoint_dir):
            self._replace(checkpoint_dir, f"{self.bucket_path}/checkpoint-best")

    @classmethod
    def download_latest_checkpoint(cls, local_dir, model_name, bucket=HF_BUCKET):
        """
        Downloads <bucket>/<model_name>/checkpoint-latest into local_dir and
        returns the local path. Returns None if no latest checkpoint exists.
        """
        fs = HfFileSystem()
        remote_path = f"{bucket}/{model_name}/checkpoint-latest"
        if not fs.exists(remote_path):
            return None

        local_checkpoint_dir = os.path.join(local_dir, "checkpoint-latest")
        os.makedirs(local_dir, exist_ok=True)

        print(f"Downloading {remote_path} -> {local_checkpoint_dir}")
        fs.get(remote_path, local_checkpoint_dir, recursive=True)

        return local_checkpoint_dir
