import os
import boto3
from pathlib import Path
from transformers import TrainerCallback


S3_BUCKET = "amzn-s3-bucket-564290535581-us-east-2-an"


class S3UploadCallback(TrainerCallback):
    """
    Uploads the best checkpoint to S3 after it is saved, and provides a
    classmethod to download the latest checkpoint back to a local directory
    for resuming training on a fresh instance.
    Relies on AWS credentials from environment variables:
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
    or an IAM role attached to the instance.
    """

    def __init__(self, s3_prefix="checkpoints"):
        self.s3_prefix = s3_prefix
        self.s3 = boto3.client("s3")

    @classmethod
    def download_latest_checkpoint(cls, local_dir, s3_prefix="checkpoints"):
        """
        Lists checkpoints under s3_prefix in S3, downloads the one with the
        highest step number to local_dir, and returns the local path.
        Returns None if no checkpoints exist in S3.
        """
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{s3_prefix}/checkpoint-")

        checkpoint_steps = set()
        for page in pages:
            for obj in page.get("Contents", []):
                # key looks like: checkpoints/checkpoint-500/pytorch_model.bin
                parts = obj["Key"].split("/")
                if len(parts) >= 2 and parts[1].startswith("checkpoint-"):
                    try:
                        step = int(parts[1].split("-")[1])
                        checkpoint_steps.add(step)
                    except ValueError:
                        pass

        if not checkpoint_steps:
            return None

        best_step = max(checkpoint_steps)
        checkpoint_s3_prefix = f"{s3_prefix}/checkpoint-{best_step}"
        local_checkpoint_dir = os.path.join(local_dir, f"checkpoint-{best_step}")
        os.makedirs(local_checkpoint_dir, exist_ok=True)

        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=checkpoint_s3_prefix + "/")
        for page in pages:
            for obj in page.get("Contents", []):
                relative_path = obj["Key"][len(checkpoint_s3_prefix) + 1:]
                local_path = os.path.join(local_checkpoint_dir, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                print(f"Downloading s3://{S3_BUCKET}/{obj['Key']} -> {local_path}")
                s3.download_file(S3_BUCKET, obj["Key"], local_path)

        return local_checkpoint_dir

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(checkpoint_dir):
            return

        self._upload_dir(checkpoint_dir)

    def _upload_dir(self, local_dir):
        for path in Path(local_dir).rglob("*"):
            if path.is_file():
                s3_key = f"{self.s3_prefix}/{path.relative_to(Path(local_dir).parent)}"
                print(f"Uploading {path} -> s3://{S3_BUCKET}/{s3_key}")
                self.s3.upload_file(str(path), S3_BUCKET, s3_key)
