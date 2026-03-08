from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tevatron.retriever.modeling import DenseModel
from transformers import AutoTokenizer

DEFAULT_BASE_MODEL = "Qwen/Qwen3-Embedding-4B"
TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_LORA_PATH = TRAINING_DIR / "models" / "AgentIR-4B-lora"
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "models" / "AgentIR-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the trained AgentIR LoRA adapter into a standalone checkpoint."
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model used during training.",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=DEFAULT_LORA_PATH,
        help="Path to the trained LoRA adapter checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the merged standalone checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.lora_path.exists():
        raise FileNotFoundError(
            f"LoRA checkpoint not found at {args.lora_path}. "
            "Run training/train.sh first or pass --lora-path explicitly."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading base model {args.base_model} with adapter {args.lora_path}...")
    model = DenseModel.load(
        args.base_model,
        pooling="last",
        normalize=True,
        lora_name_or_path=str(args.lora_path),
        torch_dtype=dtype,
    ).to(device).eval()

    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")

    print(f"Saving merged checkpoint to {args.output_dir}...")
    model.encoder.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Merged checkpoint written to {args.output_dir}")


if __name__ == "__main__":
    main()
