from sadra.manager import SADRAManager

# 1. Az önceki analizden elde ettiğin gerçek veriler (Örnek)
# Not: Classifier'ı çıkardım çünkü LoRA genelde Transformer bloklarına uygulanır.
mock_scores = {
    "distilbert.transformer.layer.0.ffn.lin1.weight": 2.5241,
    "distilbert.embeddings.word_embeddings.weight": 1.4103,
    "distilbert.transformer.layer.0.attention.q_lin.weight": 0.2757,
    "distilbert.transformer.layer.0.attention.k_lin.weight": 0.2296,
    "distilbert.transformer.layer.0.attention.out_lin.weight": 0.1549,
    "distilbert.transformer.layer.5.attention.k_lin.weight": 0.0644,
    "distilbert.transformer.layer.2.attention.q_lin.weight": 0.0291,
    # Biraz daha yapay veri ekleyelim ki dağılımı görelim
    "distilbert.transformer.layer.3.attention.v_lin.weight": 0.1000,
    "distilbert.transformer.layer.4.ffn.lin2.weight": 1.8000,
}

# 2. Yöneticiyi Başlat
# Hedef: Ortalama rank 8 olsun istiyoruz. 
# Toplam Bütçe = Katman Sayısı (9) * 8 = 72 puan
num_layers = len(mock_scores)
target_avg_rank = 8
total_budget = num_layers * target_avg_rank

manager = SADRAManager(total_rank_budget=total_budget, min_rank=4, max_rank=32)

# 3. Rankları Hesapla
print("--- SADRA Rank Dağıtımı Hesaplanıyor ---")
rank_config = manager.allocate_ranks(mock_scores)

# 4. Sonuçları Yazdır
print("\n--- ÖNERİLEN YAPILANDIRMA ---")
for layer, rank in sorted(rank_config.items(), key=lambda x: x[1], reverse=True):
    print(f"Rank: {rank:<3} | Layer: {layer}")