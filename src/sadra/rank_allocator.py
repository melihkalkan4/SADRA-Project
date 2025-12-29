# src/sadra/rank_allocator.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class RankAllocConfig:
    """
    SADRA Bütçe ve Rank Tahsisat Yapılandırması.
    """
    lora_budget: int = 600_000        # Hedeflenen toplam LoRA parametre sayısı
    rank_min: int = 2                 # Budanmayan katmanlar için minimum rank
    rank_max: int = 32                # En hassas katmanlar için maksimum rank
    allowed_ranks: Tuple[int, ...] = (0, 2, 4, 8, 16, 32) # Kesikli rank kümesi
    power: float = 1.0                # Hassasiyet farklarını keskinleştirme katsayısı
    eps: float = 1e-12

def allocate_ranks(
    sensitivity_scores: Dict[str, float],
    linear_shapes: Dict[str, Tuple[int, int]],
    cfg: RankAllocConfig
) -> Dict[str, int]:
    """
    Hessian skorlarını Impact/Cost oranına göre optimize ederek ranklara dönüştürür.
    """
    # 1. İsim normalizasyonu ve ortak anahtar kontrolü
    keys = [k for k in sensitivity_scores.keys() if k.replace(".weight", "") in linear_shapes]
    
    # 2. Hassasiyet skorlarını dönüştür ve normalize et
    # Power transform, önemli katmanları daha fazla öne çıkarır.
    vals = [max(0, sensitivity_scores[k])**cfg.power for k in keys]
    smin, smax = min(vals), max(vals)
    
    init_ranks = {}
    for k in keys:
        # 0-1 arasına normalize et
        norm_s = (vals[keys.index(k)] - smin) / (smax - smin + cfg.eps)
        
        # Sürekli (continuous) rank hesapla
        r_cont = cfg.rank_min + norm_s * (cfg.rank_max - cfg.rank_min)
        
        # En yakın 'allowed_ranks' değerine yuvarla (Quantization)
        r = min(cfg.allowed_ranks, key=lambda a: abs(a - r_cont))
        init_ranks[k] = int(r)

    # 3. İteratif Bütçe Optimizasyonu (Impact/Cost Ratio)
    def get_total_cost(ranks):
        # Maliyet = rank * (in_dim + out_dim)
        return sum(r * sum(linear_shapes[k.replace(".weight", "")]) for k, r in ranks.items())

    current_cost = get_total_cost(init_ranks)
    
    # Eğer bütçe aşılıyorsa, en 'verimsiz' katmanlardan rank düşürmeye başla
    if current_cost > cfg.lora_budget:
        # Ratio = Duyarlılık / Maliyet. Düşük oranlı katmanlar önce budanır.
        items = sorted([
            (sensitivity_scores[k] / ((sum(linear_shapes[k.replace(".weight", "")]) * init_ranks[k]) + cfg.eps), k) 
            for k in keys if init_ranks[k] > 0
        ])
        
        allowed = sorted(cfg.allowed_ranks)
        while current_cost > cfg.lora_budget:
            for _, k in items:
                r = init_ranks[k]
                if r <= 0: continue
                
                # Mevcut rank'ı kümeden bir alt seviyeye indir
                # Örn: 16 -> 8, 8 -> 4...
                idx = allowed.index(r)
                new_r = allowed[idx - 1] if idx > 0 else 0
                
                init_ranks[k] = new_r
                current_cost = get_total_cost(init_ranks)
                
                if current_cost <= cfg.lora_budget:
                    break
            else:
                break # Daha fazla düşürülecek rank kalmadı

    print(f"[SADRA] Allocation complete. Final Cost: {current_cost:,} / Budget: {cfg.lora_budget:,}")
    return init_ranks