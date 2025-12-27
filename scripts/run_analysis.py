import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
from sadra.core import estimate_hessian_trace
import sys

# Cihaz seçimi (GPU varsa kullan, yoksa CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"SADRA Analysis Script başlatılıyor... Cihaz: {device.upper()}")

# 1. MODEL VE TOKENIZER HAZIRLIĞI
# Hızlı sonuç almak için 'distilbert' kullanıyoruz. 
# Gerçek makalede 'roberta-large' veya 'llama-3' kullanacağız.
model_id = "distilbert-base-uncased"
print(f"Model indiriliyor: {model_id}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
except Exception as e:
    print(f"Hata: Model indirilemedi. İnternet bağlantını kontrol et.\nDetay: {e}")
    sys.exit(1)

# 2. VERİ SETİ HAZIRLIĞI (GLUE / SST-2)
print("Veri seti indiriliyor (GLUE/SST-2)...")
raw_datasets = load_dataset("glue", "sst2", split="train[:100]") # Sadece ilk 100 örnek (Hız için)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=False)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Data Collator (Batch'leri aynı boyuta getirmek için)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
    tokenized_datasets, 
    shuffle=True, 
    batch_size=8, 
    collate_fn=data_collator
)

# 3. SADRA ANALİZİ (Hessian Trace Estimation)
print("\n--- SADRA: Katman Duyarlılık Analizi Başlıyor ---")
#'core.py' fonksiyonunu çağırıyoruz:
scores = estimate_hessian_trace(
    model=model,
    data_loader=train_dataloader,
    device=device,
    num_batches=5,  # 5 batch (40 örnek) üzerinden hesapla
    num_vectors=10  # Her parametre için 10 rastgele vektör
)

# 4. SONUÇLARI GÖRSELLEŞTİRME
print("\n--- SONUÇLAR: En Duyarlı 10 Katman ---")
# Skorlara göre büyükten küçüğe sırala
sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

for name, score in sorted_scores[:10]:
    print(f"Layer: {name:<50} | Sensitivity (Trace): {score:.4f}")

print("\nAnaliz Başarıyla Tamamlandı! 🚀")