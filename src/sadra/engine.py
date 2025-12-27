import torch
from peft import get_peft_model, LoraConfig, TaskType

def apply_sadra_to_model(model, rank_config, default_rank=8, alpha_ratio=2.0):
    """
    SADRA (Sensitivity-Aware Dynamic Rank Allocation) motoru.
    
    Bu fonksiyon, hesaplanan hassasiyet skorlarına göre her katmana 
    farklı Rank (r) atayarak PEFT modelini oluşturur.
    
    Args:
        model: HuggingFace Base Model (Pre-trained)
        rank_config: {layer_name: int_rank} sözlüğü (Manager'dan gelen)
        default_rank: Config'de olmayan katmanlar için varsayılan rank.
        alpha_ratio: LoRA Alpha = Rank * Ratio (Genelde 2x kararlıdır).
    
    Returns:
        peft_model: Eğitime hazır SADRA modeli.
    """
    print(f"\n[SADRA] Motor Başlatılıyor... Hedef Katman Sayısı: {len(rank_config)}")
    
    # 1. Rank Pattern ve Target Modules Hazırlığı
    # PEFT kütüphanesi, hangi modüllere LoRA takılacağını 'target_modules' listesiyle,
    # hangi katmana kaç rank verileceğini 'rank_pattern' sözlüğüyle anlar.
    
    rank_pattern = {}
    target_suffixes = set()
    
    for name, rank in rank_config.items():
        # Parametre isminden (weight) kurtul, modül ismini al
        # Örn: 'distilbert.layer.0.lin1.weight' -> 'distilbert.layer.0.lin1'
        clean_name = name.replace(".weight", "")
        
        # Pattern'e ekle
        rank_pattern[clean_name] = rank
        
        # Soneki (Suffix) bul (lin1, q_lin, query, key, value vb.)
        # PEFT'in 'target_modules' parametresi için gereklidir.
        suffix = clean_name.split(".")[-1]
        target_suffixes.add(suffix)

    print(f"[SADRA] Tespit Edilen Modül Tipleri: {list(target_suffixes)}")
    
    # 2. Dinamik Konfigürasyonun Oluşturulması
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, # Sınıflandırma görevi (Modeline göre değişebilir)
        inference_mode=False,
        r=default_rank,             # Varsayılan (Fallback) değer
        lora_alpha=default_rank * alpha_ratio, # Alpha genelde rank'ın 2 katıdır
        lora_dropout=0.1,
        target_modules=list(target_suffixes), # ['q_lin', 'lin1'...]
        rank_pattern=rank_pattern   # <--- İŞTE SADRA BURADA DEVREYE GİRİYOR
    )
    
    # 3. Modelin Dönüştürülmesi
    try:
        peft_model = get_peft_model(model, peft_config)
        
        # Başarı İstatistiği
        trainable_params, all_params = peft_model.get_nb_trainable_parameters()
        print(f"[SADRA] Model Hazır! 🚀")
        print(f" -> Eğitilebilir Parametre: {trainable_params:,}")
        print(f" -> Oran: %{100 * trainable_params / all_params:.2f}")
        
        return peft_model
        
    except Exception as e:
        print(f"[SADRA ERROR] Model dönüştürülürken kritik hata: {e}")
        raise e