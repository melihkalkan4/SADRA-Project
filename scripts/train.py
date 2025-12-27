import os
import torch
import numpy as np
import evaluate
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
from sadra.core import estimate_hessian_trace
from sadra.manager import SADRAManager
from sadra.engine import apply_sadra_to_model

# --- AYARLAR ---
MODEL_ID = "distilbert-base-uncased"
TASK = "sst2" # GLUE Benchmark görevi
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./output/sadra-run-1"

def main():
    print(f"--- SADRA EXPERIMENT BAŞLIYOR ({DEVICE.upper()}) ---")
    
    # 1. Veri ve Model Hazırlığı
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset("glue", TASK)
    
    def tokenize_fn(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=128)
    
    tokenized_ds = dataset.map(tokenize_fn, batched=True)
    
    # PADDING MEKANİZMASI
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Model Yükle
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2).to(DEVICE)
    
    # Analiz için DataLoader
    from torch.utils.data import DataLoader
    analysis_loader = DataLoader(
        tokenized_ds["train"].remove_columns(["sentence", "idx"]).rename_column("label", "labels").with_format("torch"),
        batch_size=8,
        shuffle=True,
        collate_fn=data_collator 
    )

    # ---------------------------------------------------------
    # ADIM 1: SENSITIVITY ANALYSIS (Hessian Trace)
    # ---------------------------------------------------------
    print("\n[Phase 1] Katman Duyarlılık Analizi...")
    sensitivity_scores = estimate_hessian_trace(
        model=model, 
        data_loader=analysis_loader, 
        device=DEVICE,
        num_batches=10, 
        num_vectors=10
    )
    
    # ---------------------------------------------------------
    # ADIM 2: DYNAMIC RANK ALLOCATION
    # ---------------------------------------------------------
    print("\n[Phase 2] Rank Bütçesi Dağıtılıyor...")
    avg_rank = 8
    num_layers = len([k for k in sensitivity_scores if "weight" in k]) 
    total_budget = num_layers * avg_rank
    
    manager = SADRAManager(total_rank_budget=total_budget, min_rank=2, max_rank=32)
    rank_config = manager.allocate_ranks(sensitivity_scores)
    
    # ---------------------------------------------------------
    # ADIM 3: SADRA INJECTION
    # ---------------------------------------------------------
    print("\n[Phase 3] Dinamik Model Oluşturuluyor...")
    model.cpu() 
    sadra_model = apply_sadra_to_model(model, rank_config)
    sadra_model.to(DEVICE)
    
    # ---------------------------------------------------------
    # ADIM 4: TRAINING (Fine-Tuning)
    # ---------------------------------------------------------
    print("\n[Phase 4] Eğitim Başlıyor...")
    
    metric = evaluate.load("glue", TASK)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-4, 
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch", # <--- DÜZELTİLEN KISIM BURASI (Eski: evaluation_strategy)
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50, 
        report_to="none"
    )
    
    trainer = Trainer(
        model=sadra_model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    print("\n--- SADRA DENEY SONUÇLARI ---")
    eval_results = trainer.evaluate()
    print(eval_results)

if __name__ == "__main__":
    main()