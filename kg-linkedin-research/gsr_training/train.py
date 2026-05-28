"""
train.py
Fine-tune T5 on LinkedIn KG relation-chain prediction (GSR style).

Input:  processed_data/linkedin_train/  and  linkedin_val/
Output: trained_models/linkedin-gsr-<model_id>/

This is a direct adaptation of GSR's pretrain.py, stripped of the
WebQSP/CWQ dataset loading and wired to our LinkedIn HF datasets.

Usage (from repo root):
    uv run kg-linkedin-research/gsr_training/train.py \\
        --model_id t5-small \\
        --num_epochs 20 \\
        --bsz 32 \\
        --lr 5e-4

For a smoke test (1 epoch, verify pipeline end-to-end):
    uv run kg-linkedin-research/gsr_training/train.py \\
        --model_id t5-small --num_epochs 1 --bsz 8 --smoke_test
"""

import os
from argparse import ArgumentParser
from pathlib import Path

import evaluate
import numpy as np
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

HERE = Path(__file__).parent
TOKENIZER_DIR = HERE / "tokenizer" / "t5-linkedin-rel"
DATA_DIR = HERE / "processed_data"
OUTPUT_DIR = HERE / "trained_models"


def run_train(args):
    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    train_dataset = load_from_disk(str(DATA_DIR / "linkedin_train"))
    eval_dataset  = load_from_disk(str(DATA_DIR / "linkedin_val"))

    if args.smoke_test:
        train_dataset = train_dataset.select(range(min(50, len(train_dataset))))
        eval_dataset  = eval_dataset.select(range(min(20, len(eval_dataset))))
        print(f"[smoke_test] trimmed to {len(train_dataset)} train / {len(eval_dataset)} eval")

    print(f"Train: {len(train_dataset)} | Val: {len(eval_dataset)}")

    # ------------------------------------------------------------------
    # Model + tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
    model.resize_token_embeddings(len(tokenizer))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    em_metric = evaluate.load("exact_match")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = em_metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"exact_match": round(result["exact_match"], 4)}

    # ------------------------------------------------------------------
    # Training args
    # ------------------------------------------------------------------
    run_name = f"linkedin-gsr-{args.model_id.replace('/', '-')}"
    if args.smoke_test:
        run_name += "-smoke"
    out_dir = str(OUTPUT_DIR / run_name)

    # bf16 only when explicitly requested (requires CUDA BF16 support)
    use_bf16 = args.bf16
    use_fp16 = args.fp16

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_dir=f"{out_dir}/logs",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        predict_with_generate=True,
        report_to="none",       # disable wandb / mlflow by default
        bf16=use_bf16,
        fp16=use_fp16,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    model.config.use_cache = False  # silence warning during training

    trainer.train()

    # Save final model + tokenizer together
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nModel saved to {out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", default="t5-small",
                        help="HuggingFace model ID (t5-small, t5-base, ...)")
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--bf16", action="store_true",
                        help="Enable bfloat16 (requires CUDA BF16 capable GPU)")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable float16 (requires CUDA)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Trim to 50 train / 20 val samples for quick pipeline check")
    args = parser.parse_args()

    run_train(args)
