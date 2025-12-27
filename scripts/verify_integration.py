import torch
from transformers import AutoModelForSequenceClassification
from sadra.engine import apply_sadra_to_model

# 1. Modeli Yükle
print("Model yükleniyor...")
model_id = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# 2. Yapay Bir Rank Config Hazırla (Sanki analizden gelmiş gibi)
# Engine kodumuzun '.weight' silme özelliğini test etmek için tam isimleri kullanıyoruz.
mock_rank_config = {
    "distilbert.transformer.layer.0.ffn.lin1.weight": 32,  # Yüksek Rank
    "distilbert.transformer.layer.0.attention.q_lin.weight": 4,   # Düşük Rank
}

# 3. SADRA Engine Çalıştır
peft_model = apply_sadra_to_model(model, mock_rank_config)

# 4. Parametre Sayısını Kontrol Et
print("\n--- Model Özeti ---")
peft_model.print_trainable_parameters()