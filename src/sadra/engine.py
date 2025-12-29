# src/sadra/engine.py
from __future__ import annotations
from typing import Dict, Any, List
from peft import get_peft_model, LoraConfig, TaskType

def apply_sadra_to_model(
    model: Any,
    rank_config: Dict[str, int],
    task_type: TaskType = TaskType.SEQ_CLS,
    alpha_ratio: float = 2.0,
    lora_dropout: float = 0.1,
    bias: str = "none",
) -> Any:
    """
    SADRA Engine: Belirlenmiş rank_config'i modele dinamik olarak uygular.
    
    Özellikler:
    - Sadece rank > 0 olan katmanlara LoRA ekler (Otomatik Budama).
    - Her katman için (rank * alpha_ratio) formülüyle dinamik alpha atar.
    - PEFT çakışmalarını önlemek için modules_to_save kontrolü içerir.
    """
    
    if not isinstance(rank_config, dict) or len(rank_config) == 0:
        raise ValueError("[SADRA] rank_config is empty or invalid.")

    rank_pattern = {}
    alpha_pattern = {}
    target_modules = []
    skipped_zero = 0

    # 1. Rank ve Alpha Pattern Hazırlığı
    for name, r in rank_config.items():
        # İsim normalizasyonu: parametre isminden (.weight) modül ismine geçiş
        clean_name = name.replace(".weight", "").strip()
        
        val = int(r)
        if val > 0:
            rank_pattern[clean_name] = val
            # Matematiksel Ölçekleme: r * alpha_ratio
            # Bu, farklı ranklardaki katmanların gradyan akışını sabit tutar.
            alpha_pattern[clean_name] = int(val * alpha_ratio)
            target_modules.append(clean_name)
        else:
            skipped_zero += 1

    if not target_modules:
        raise ValueError("[SADRA] No modules with r > 0 found. All layers might have been pruned.")

    # 2. LoRA Yapılandırması
    peft_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=8,              # Fallback rank
        lora_alpha=16,    # Fallback alpha
        lora_dropout=float(lora_dropout),
        bias=bias,
        # SADECE hassas bulunan katmanları hedefle
        target_modules=target_modules,
        # Dinamik rank ve alpha dağılımı
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        # KRİTİK HATA ÇÖZÜMÜ: LoRA katmanlarının kaydedilme çakışmasını önler
        modules_to_save=None 
    )

    # 3. PEFT Modelini Oluştur ve Döndür
    try:
        peft_model = get_peft_model(model, peft_config)
        
        # Raporlama
        trainable_params, all_params = peft_model.get_nb_trainable_parameters()
        print("\n" + "="*50)
        print("[SADRA ENGINE] Successfully Initialized")
        print(f"[*] Target Modules (r>0): {len(target_modules)}")
        print(f"[*] Pruned Modules (r=0): {skipped_zero}")
        print(f"[*] Trainable Ratio: %{100 * trainable_params / all_params:.2f}")
        print("="*50 + "\n")
        
        return peft_model
    except Exception as e:
        print(f"[SADRA ERROR] Failed to initialize PEFT model: {e}")
        raise

def preview_sadra_allocation(rank_config: Dict[str, int]):
    """Tahsis edilen rankların özetini yazdırır."""
    sorted_ranks = sorted(rank_config.items(), key=lambda x: x[1], reverse=True)
    print("\n[SADRA] Top-10 Allocated Ranks:")
    for name, r in sorted_ranks[:10]:
        print(f"  - {name}: r={r}")