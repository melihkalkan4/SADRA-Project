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

# --- Benchmark Listesi ---
TASKS = ["sst2", "mrpc", "qnli"] 
MODEL_ID = "distilbert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GLUE görevlerinin sütun isimleri haritası
GLUE_TASK_KEYS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"), # QNLI burada patlıyordu, düzelttik
    "mnli": ("premise", "hypothesis"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
}

def run_experiment(task_name):
    print(f"\n{'='*40}")
    print(f"🚀 BAŞLATILIYOR: GLUE/{task_name.upper()}")
    print(f"{'='*40}")
    
    # 1. Dataset ve Metrik Yükle
    dataset = load_dataset("glue", task_name)
    metric = evaluate.load("glue", task_name)
    
    # Sütun isimlerini belirle
    sentence1_key, sentence2_key = GLUE_TASK_KEYS[task_name]
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    def tokenize_fn(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, max_length=128)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, max_length=128)
    
    tokenized_ds = dataset.map(tokenize_fn, batched=True)
    
    # KRİTİK DÜZELTME: 'label' sütununu 'labels' yap (PyTorch için zorunlu)
    if "label" in tokenized_ds["train"].column_names:
        tokenized_ds = tokenized_ds.rename_column("label", "labels")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Model Yükle
    num_labels = 3 if task_name.startswith("mnli") else 2
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=num_labels).to(DEVICE)
    
    # --- PHASE 1: ANALİZ ---
    from torch.utils.data import DataLoader
    
    # Analiz için sütun temizliği
    # Sadece modelin anlayacağı sütunları bırakıyoruz
    keep_cols = ["input_ids", "attention_mask", "labels", "token_type_ids"]
    analysis_subset = tokenized_ds["train"].select(range(200)) # İlk 200 örnek
    
    # Dataset'te olmayan sütunları filtrele (bazı modellerde token_type_ids yoktur)
    available_cols = [c for c in keep_cols if c in analysis_subset.column_names]
    
    # Remove columns yerine sadece gerekli olanları seçip formatlıyoruz
    analysis_subset = analysis_subset.select_columns(available_cols)
    analysis_subset.set_format("torch")

    analysis_loader = DataLoader(
        analysis_subset,
        batch_size=8, 
        shuffle=True, 
        collate_fn=data_collator
    )
    
    print(f"[{task_name}] Hessian Analizi...")
    try:
        scores = estimate_hessian_trace(model, analysis_loader, DEVICE, num_batches=10, num_vectors=10)
    except Exception as e:
        print(f"Hessian Analizi Hatası: {e}")
        raise e
    
    # --- PHASE 2: RANKLAMA ---
    avg_rank = 8
    target_layers = [k for k in scores if "weight" in k]
    if not target_layers:
        print("Uyarı: LoRA uygulanacak katman bulunamadı, varsayılanlar kullanılıyor.")
        # Fallback (Hata durumunda boş dönmesin diye)
        total_budget = 36 * 8
    else:
        total_budget = len(target_layers) * avg_rank
    
    manager = SADRAManager(total_rank_budget=total_budget, min_rank=2, max_rank=32)
    rank_config = manager.allocate_ranks(scores)
    
    # --- PHASE 3: EĞİTİM ---
    model.cpu()
    sadra_model = apply_sadra_to_model(model, rank_config)
    sadra_model.to(DEVICE)
    
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir=f"./output/sadra_{task_name}",
        learning_rate=2e-4,
        per_device_train_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        logging_steps=50,
        save_total_limit=1 # Disk dolmasın diye sadece son modeli tut
    )
    
    trainer = Trainer(
        model=sadra_model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"], 
        processing_class=tokenizer, # Yeni versiyonlarda tokenizer yerine bu önerilir
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    result = trainer.evaluate()
    
    # Anahtar metriği bul (accuracy veya f1)
    main_score = result.get("eval_accuracy") or result.get("eval_f1") or result.get("eval_matthews_correlation")
    return main_score

# --- ANA DÖNGÜ ---
if __name__ == "__main__":
    results = {}
    for task in TASKS:
        try:
            score = run_experiment(task)
            results[task] = score
        except Exception as e:
            print(f"HATA ({task}): {e}")
            import traceback
            traceback.print_exc()
            results[task] = "FAILED"
            
    print("\n" + "="*40)
    print("🏆 FINAL BENCHMARK RESULTS")
    print("="*40)
    for t, s in results.items():
        print(f"Task: {t.upper():<10} | Score: {s}")