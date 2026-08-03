import numpy as np
from datasets import load_from_disk
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger

log = get_logger(__name__)

# Must match prepare_ner_data.py
LABELS = [
    "O",
    "B-EXPERIMENT",
    "I-EXPERIMENT",
    "B-MATERIAL",
    "I-MATERIAL",
    "B-DEVICE",
    "I-DEVICE",
    "B-VALUE",
    "I-VALUE",
]
label2id = {lbl: i for i, lbl in enumerate(LABELS)}
id2label = {i: lbl for i, lbl in enumerate(LABELS)}


def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"], truncation=True, max_length=512, is_split_into_words=True
    )
    all_labels = []
    for i, tags in enumerate(examples["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids, previous_word_idx = [], None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[tags[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2label[p] for p, lbl in zip(pred, label, strict=False) if lbl != -100]
        for pred, label in zip(predictions, labels, strict=False)
    ]
    true_labels = [
        [id2label[lbl] for p, lbl in zip(pred, label, strict=False) if lbl != -100]
        for pred, label in zip(predictions, labels, strict=False)
    ]
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def fine_tune(
    model_name: str, dataset, output_dir: str, batch_size: int = 2, grad_accum: int = 4
) -> dict:
    # batch_size=8, not 16 -- deliberately chosen to comfortably fit a 6GB laptop GPU.
    log.info(f"Fine-tuning {model_name} -> {output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )
    train_tok = dataset["train"].map(
        lambda e: tokenize_and_align_labels(e, tokenizer), batched=True
    )
    val_tok = dataset["validation"].map(
        lambda e: tokenize_and_align_labels(e, tokenizer), batched=True
    )
    test_tok = dataset["test"].map(lambda e: tokenize_and_align_labels(e, tokenizer), batched=True)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="no",  # only save once manually at the end
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,  # effective batch = 2*4 = 8
        num_train_epochs=10,
        load_best_model_at_end=False,  # requires save_strategy != "no"
        fp16=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_results = trainer.evaluate(test_tok)
    trainer.save_model(output_dir)
    log.info(f"{model_name} test F1: {test_results['eval_f1']:.4f}")
    return test_results


if __name__ == "__main__":
    ds = load_from_disk("data/ner_processed")

    matscibert_results = fine_tune("m3rg-iitd/matscibert", ds, "models/matscibert_ner")
    bert_results = fine_tune("bert-base-uncased", ds, "models/bert_baseline_ner")

    log.info("=== FINAL COMPARISON ===")
    log.info(f"MatSciBERT test F1:   {matscibert_results['eval_f1']:.4f}")
    log.info(f"Generic BERT test F1: {bert_results['eval_f1']:.4f}")
