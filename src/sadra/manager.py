import torch
import numpy as np

class SADRAManager:
    def __init__(self, total_rank_budget, min_rank=2, max_rank=64):
        """
        SADRA (Sensitivity-Aware Dynamic Rank Allocation) Yöneticisi.
        """
        self.total_budget = total_rank_budget
        self.min_rank = min_rank
        self.max_rank = max_rank

    def allocate_ranks(self, sensitivity_scores):
        """
        Duyarlılık skorlarına (Hessian Trace) göre bütçeyi dağıtır.
        """
        # 1. Sadece LoRA uygulanabilir katmanları filtrele
        target_layers = {
            k: v for k, v in sensitivity_scores.items() 
            if "attention" in k or "ffn" in k or "lin" in k
        }
        
        if not target_layers:
            print("Uyarı: LoRA uygulanabilecek katman bulunamadı.")
            return {}

        # İsimleri ve skorları ayır
        layer_names = list(target_layers.keys())
        scores = np.array(list(target_layers.values()))
        
        # 2. Skorları Normalize Et
        total_score = np.sum(scores) + 1e-6
        normalized_scores = scores / total_score
        
        # 3. Ham Rankları Hesapla
        raw_ranks = normalized_scores * self.total_budget
        
        # 4. Integer'a Çevir ve Sınırla
        allocated_ranks = np.clip(np.round(raw_ranks), self.min_rank, self.max_rank)
        
        # --- KRİTİK DÜZELTME BURADA ---
        # NumPy int64 türünü standart Python int türüne çeviriyoruz.
        # JSON serileştirme hatasını (TypeError) çözen satır budur:
        rank_config = {
            name: int(rank) for name, rank in zip(layer_names, allocated_ranks)
        }
        
        # --- İstatistikler ---
        print(f"\n[SADRA Allocation Report]")
        print(f"Toplam Hedef Katman: {len(layer_names)}")
        print(f"En Yüksek Rank: {max(rank_config.values())}")
        print(f"En Düşük Rank: {min(rank_config.values())}")
        
        # İstatistik için numpy mean kullanabiliriz ama config saf int kalmalı
        avg_r = sum(rank_config.values()) / len(rank_config)
        print(f"Ortalama Rank: {avg_r:.2f}")
        
        return rank_config