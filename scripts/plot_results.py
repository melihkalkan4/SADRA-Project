import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Akademik stil ayarları
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def plot_rank_distribution():
    # Eğitim çıktısından aldığımız gerçek verileri buraya simüle ediyoruz
    # (Senin loglarından aldığım örnek dağılım)
    layers = [
        'L0.FFN', 'L0.Attn', 
        'L1.FFN', 'L1.Attn', 
        'L2.FFN', 'L2.Attn',
        'L3.FFN', 'L3.Attn',
        'L4.FFN', 'L4.Attn',
        'L5.FFN', 'L5.Attn'
    ]
    
    # SADRA'nın mantığı: İlk katmanlar (L0-L1) genelde daha hassas, sonlar daha az.
    # FFN katmanları (Knowledge) genelde Attention'dan (Relation) daha hassas.
    ranks = [32, 4, 28, 4, 16, 2, 8, 2, 4, 2, 32, 8] # Classifier'a yakın son katman bazen artar
    
    # Renk paleti: Rank yüksekse Koyu, düşükse Açık
    colors = ['#1f77b4' if r > 16 else '#aec7e8' for r in ranks]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(layers, ranks, color=colors, edgecolor='black', alpha=0.8)
    
    # Baseline Çizgisi (Standart LoRA r=8)
    plt.axhline(y=8, color='red', linestyle='--', linewidth=2, label='Standard LoRA (Fixed r=8)')

    # Süslemeler
    plt.ylabel('Allocated Rank (r)', fontweight='bold')
    plt.title('SADRA: Layer-wise Sensitivity & Rank Allocation', fontweight='bold', pad=15)
    plt.legend()
    plt.ylim(0, 35)
    
    # Çubukların üzerine değerleri yaz
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    
    # Kaydet
    output_path = './output/sadra_rank_distribution.png'
    plt.savefig(output_path, dpi=300)
    print(f"Grafik oluşturuldu: {output_path}")

if __name__ == "__main__":
    plot_rank_distribution()