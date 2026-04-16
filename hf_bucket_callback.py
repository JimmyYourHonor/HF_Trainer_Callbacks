import os
from pathlib import Path
from huggingface_hub import HfFileSystem
from transformers import TrainerCallback


HF_BUCKET = "hf://buckets/JimmyFu/FeatureDetectionAndDescriptionCheckpoint"


class HFBucketCallback(TrainerCallback):
    """
    Uploads checkpoints to a HuggingFace Storage Bucket after each save,
    and provides a classmethod to download the latest checkpoint for resuming
    training on a fresh instance.
    Relies on HF_TOKEN environment variable or a prior `huggingface-cli login`.
    """

    def __init__(self, bucket=HF_BUCKET):
        self.bucket = bucket
        self.fs = HfFileSystem()

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(checkpoint_dir):
            return

        remote_path = f"{self.bucket}/checkpoint-{state.global_step}"
        print(f"Uploading {checkpoint_dir} -> {remote_path}")
        self.fs.put(checkpoint_dir, remote_path, recursive=True)

    @classmethod
    def download_latest_checkpoint(cls, local_dir, bucket=HF_BUCKET):
        """
        Lists checkpoints in the bucket, downloads the one with the highest
        step number to local_dir, and returns the local path.
        Returns None if no checkpoints exist in the bucket.
        """
        fs = HfFileSystem()

        try:
            entries = fs.ls(bucket, detail=False)
        except Exception:
            return None

        checkpoint_steps = []
        for entry in entries:
            name = Path(entry).name
            if name.startswith("checkpoint-"):
                try:
                    checkpoint_steps.append(int(name.split("-")[1]))
                except ValueError:
                    pass

        if not checkpoint_steps:
            return None

        best_step = max(checkpoint_steps)
        remote_path = f"{bucket}/checkpoint-{best_step}"
        local_checkpoint_dir = os.path.join(local_dir, f"checkpoint-{best_step}")
        os.makedirs(local_checkpoint_dir, exist_ok=True)

        print(f"Downloading {remote_path} -> {local_checkpoint_dir}")
        fs.get(remote_path, local_checkpoint_dir, recursive=True)

        return local_checkpoint_dir
