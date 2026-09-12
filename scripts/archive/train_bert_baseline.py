from datasets import load_from_disk

from src.ner.train import fine_tune, log

ds = load_from_disk("data/ner_processed")
bert_results = fine_tune("bert-base-uncased", ds, "models/bert_baseline_ner")
log.info(f"BERT baseline test F1: {bert_results['eval_f1']:.4f}")
