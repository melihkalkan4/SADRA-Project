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

# --- SADECE QNLI AYARLARI ---
TASK_NAME = "qnli" 
MODEL_ID = "distilbert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_qnli_special():
    print(f"\n{'='*40}")
    print(f"🚑 KURTARMA OPERASYONU: GLUE/{TASK_NAME.upper()}")
    print(f"{'='*40}")
    
    # 1. Veriyi İndir
    dataset = load_dataset("glue", TASK_NAME)
    metric = evaluate.load("glue", TASK_NAME)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # QNLI Özel Tokenizer Ayarı (Soru - Cümle çifti)
    def tokenize_fn(examples):
        return tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=128)
    
    print("Veri işleniyor...")
    tokenized_ds = dataset.map(tokenize_fn, batched=True)
    
    # 'label' sütununu 'labels' yap (PyTorch kuralı)
    if "label" in tokenized_ds["train"].column_names:
        tokenized_ds = tokenized_ds.rename_column("label", "labels")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 2. Modeli Yükle
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2).to(DEVICE)
    
    # --- PHASE 1: Hessian Analizi (Düşük Bellek Modu) ---
    from torch.utils.data import DataLoader
    
    # Analiz için sadece 100 örnek alıyoruz (Hız ve bellek tasarrufu)
    analysis_subset = tokenized_ds["train"].select(range(100))
    # Sadece gerekli sütunları tut
    keep_cols = ["input_ids", "attention_mask", "labels"] 
    analysis_subset = analysis_subset.select_columns([c for c in keep_cols if c in analysis_subset.column_names])
    analysis_subset.set_format("torch")

    # Analiz Batch Size'ını 4'e düşürdük (VRAM dostu)
    analysis_loader = DataLoader(
        analysis_subset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=data_collator
    )
    
    print(f"[{TASK_NAME}] Hessian Analizi Başlıyor (Safe Mode)...")
    # Batch ve Vector sayılarını da kıstık
    scores = estimate_hessian_trace(model, analysis_loader, DEVICE, num_batches=5, num_vectors=5)
    
    # --- PHASE 2: Rank Atama ---
    # Modeldeki Linear katmanları bul
    target_layers = [k for k in scores if "weight" in k]
    if not target_layers:
        print("UYARI: Katman bulunamadı, varsayılan bütçe atanıyor.")
        total_budget = 36 * 8
    else:
        total_budget = len(target_layers) * 8 # Ortalama rank 8 olacak şekilde bütçe
    
    print(f"Hesaplanan Toplam Bütçe: {total_budget}")
    manager = SADRAManager(total_rank_budget=total_budget, min_rank=2, max_rank=32)
    rank_config = manager.allocate_ranks(scores)
    
    # --- PHASE 3: Eğitim (Düşük Batch Size) ---
    model.cpu() # GPU'yu boşalt
    sadra_model = apply_sadra_to_model(model, rank_config)
    sadra_model.to(DEVICE) # Geri yükle
    
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labels)

    # --- KRİTİK AYARLAR ---
    training_args = TrainingArguments(
        output_dir=f"./output/sadra_{TASK_NAME}_final",
        learning_rate=2e-4,
        # Batch Size'ı 32'den 16'ya düşürdük (Hafıza taşmasın diye)
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        # Her 2 adımda bir güncelleme yaparak sanal olarak 32 batch size etkisi yaratıyoruz
        gradient_accumulation_steps=2, 
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        save_total_limit=1
    )
    
    trainer = Trainer(
        model=sadra_model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"], 
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("Eğitim Başlıyor... (Bu biraz sürebilir)")
    trainer.train()
    
    print("Değerlendirme Yapılıyor...")
    result = trainer.evaluate()
    
    print(f"\n✅ SONUÇ - QNLI Accuracy: {result['eval_accuracy']}")

if __name__ == "__main__":
    run_qnli_special()