"""
ex1.py -- Fine-tune bert-base-uncased for paraphrase detection on MRPC (GLUE).

Two modes, selected by flags:
  --do_train    fine-tune a model, evaluate it on the validation split, and
                append the validation accuracy to res.txt.
  --do_predict  load a trained model and write its test-set predictions to
                predictions.txt.

Typical workflow (run --do_train once per hyper-parameter configuration so
res.txt accumulates a line for each; then run --do_predict on the best one):

  python ex1.py --do_train --num_train_epochs 5 --lr 5e-5 --batch_size 32 \
                --max_train_samples -1 --max_eval_samples -1 \
                --max_predict_samples -1 --model_path ./model_best
  python ex1.py --do_predict --model_path ./model_best --max_predict_samples -1
"""

import argparse
import os

# Avoid NaN losses on Apple Silicon if a torch op isn't yet implemented for MPS.
# Must be set BEFORE torch is imported (transformers imports torch transitively).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import wandb
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "bert-base-uncased"     # base model required by the exercise
NUM_LABELS = 2                       # MRPC labels: 0 = not paraphrase, 1 = paraphrase
RES_FILE = "res.txt"
PRED_FILE = "predictions.txt"


# --------------------------------------------------------------------------
# Command-line arguments
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MRPC paraphrase detection.")
    # -1 means "use the whole split"; otherwise keep the first n examples.
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--max_predict_samples", type=int, default=-1)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_predict", action="store_true")
    # Where the model is saved (after --do_train) and loaded from (--do_predict).
    p.add_argument("--model_path", type=str, default="./model")
    # Which split to predict on. Default is "test" (the spec). Use "validation"
    # to generate predictions needed for the qualitative-analysis question.
    p.add_argument("--predict_split", type=str, default="test",
                   choices=["test", "validation", "train"])
    return p.parse_args()


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def load_split(split, max_samples):
    """Load one MRPC split, optionally keeping only the first `max_samples`."""
    ds = load_dataset("nyu-mll/glue", "mrpc", split=split)
    if max_samples != -1:
        ds = ds.select(range(max_samples))
    return ds


def make_tokenize_fn(tokenizer):
    """Build the tokenization function for a sentence pair.

    Inputs are truncated to the model's maximum sequence length. No padding is
    applied here -- padding is done per-batch by DataCollatorWithPadding so
    each batch is padded only to its own longest example (dynamic padding).
    """
    def tokenize(batch):
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
    return tokenize


def compute_metrics(eval_pred):
    """Accuracy metric used by the Trainer for the validation split."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def do_train(args):
    set_seed(42)  # reproducible weight initialization and data shuffling

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenize = make_tokenize_fn(tokenizer)

    # The Trainer auto-drops columns the model does not accept (sentence1/2,
    # idx) and keeps the `label` column as the supervision signal.
    train_ds = load_split("train", args.max_train_samples).map(tokenize, batched=True)
    eval_ds = load_split("validation", args.max_eval_samples).map(tokenize, batched=True)

    # Default model configuration with a 2-class sequence-classification head.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    run_name = f"epochs{args.num_train_epochs}_lr{args.lr}_bs{args.batch_size}"
    wandb.init(project="mrpc-paraphrase", name=run_name, config=vars(args))

    training_args = TrainingArguments(
        output_dir="./trainer_output",
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",          # evaluate on validation each epoch
        save_strategy="no",             # do NOT save checkpoints (disk space)
        logging_strategy="steps",
        logging_steps=1,                # log the training loss every step
        report_to=["wandb"],
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,                        # transformers >= 4.46
        data_collator=DataCollatorWithPadding(tokenizer),  # dynamic padding
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Validation accuracy of the FINAL model (the exercise asks only for this).
    metrics = trainer.evaluate()
    val_acc = metrics["eval_accuracy"]
    print(f"Validation accuracy: {val_acc:.4f}")

    # Append this configuration's result to res.txt.
    # NOTE: confirm this line format matches the Moodle res.txt template.
    with open(RES_FILE, "a") as f:
        f.write(
            f"epoch_num: {args.num_train_epochs}, "
            f"lr: {args.lr}, "
            f"batch_size: {args.batch_size}, "
            f"eval_acc: {val_acc:.4f}\n"
        )

    # Save the final model and tokenizer so --do_predict can load them later.
    trainer.save_model(args.model_path)
    tokenizer.save_pretrained(args.model_path)
    print(f"Saved model to {args.model_path}")

    wandb.finish()


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def do_predict(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()  # inference mode: disables dropout and similar train-only layers

    test_ds = load_split(args.predict_split, args.max_predict_samples)
    tokenize = make_tokenize_fn(tokenizer)
    # Drop all original columns so the model receives only tokenized inputs
    # during inference (sentences and any labels are not needed by .predict()).
    tokenized_test = test_ds.map(
        tokenize, batched=True, remove_columns=test_ds.column_names
    )

    # The Trainer is used here only as a convenient batched-inference engine.
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./trainer_output",
            per_device_eval_batch_size=64,
            report_to=[],
        ),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    logits = trainer.predict(tokenized_test).predictions
    preds = np.argmax(logits, axis=-1)

    # Write predictions.txt: one line per test example, in test-set order.
    # NOTE: confirm this format matches the Moodle predictions.txt template.
    with open(PRED_FILE, "w") as f:
        for s1, s2, pred in zip(test_ds["sentence1"], test_ds["sentence2"], preds):
            f.write(f"{s1}###{s2}###{int(pred)}\n")
    print(f"Wrote {len(preds)} predictions to {PRED_FILE}")


# --------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.do_train:
        do_train(args)
    if args.do_predict:
        do_predict(args)
    if not (args.do_train or args.do_predict):
        print("Nothing to do: pass --do_train and/or --do_predict.")


if __name__ == "__main__":
    main()
