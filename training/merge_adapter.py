"""
Merge LoRA adapter into the base model for deployment.

Produces a standalone model in safetensors format that can be served
directly by vLLM without needing the adapter files at inference time.

Usage:
    python training/merge_adapter.py --adapter checkpoints/qwen3-8b-bangla-support/final_adapter
    python training/merge_adapter.py --adapter checkpoints/qwen3-8b-bangla-support/final_adapter --push mrshibly/bangla-support-qwen3-8b
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for merged model (default: {adapter}_merged)",
    )
    parser.add_argument(
        "--push",
        type=str,
        default=None,
        help="HuggingFace Hub model ID to push to (e.g., mrshibly/bangla-support-qwen3-8b)",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["q4_k_m", "q5_k_m", "q8_0"],
        help="Optional: export as GGUF with quantization for local inference",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    output_path = Path(args.output) if args.output else adapter_path.parent / "merged"

    print(f"Loading adapter from: {adapter_path}")
    print(f"Output directory: {output_path}")

    from unsloth import FastLanguageModel

    # Load the model with the LoRA adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=2048,
        load_in_4bit=False,  # Need full precision for merging
    )

    # Save merged model locally
    print("Saving merged model...")
    model.save_pretrained_merged(
        str(output_path),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"✓ Merged model saved to {output_path}")

    # Optionally push to Hub
    if args.push:
        print(f"Pushing to HuggingFace Hub: {args.push}")
        model.push_to_hub_merged(
            args.push,
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"✓ Pushed to {args.push}")

    # Optionally export as GGUF
    if args.quantize:
        gguf_path = output_path.parent / f"gguf_{args.quantize}"
        print(f"Exporting GGUF ({args.quantize}) to {gguf_path}...")
        model.save_pretrained_gguf(
            str(gguf_path),
            tokenizer,
            quantization_method=args.quantize,
        )
        print(f"✓ GGUF exported to {gguf_path}")


if __name__ == "__main__":
    main()
