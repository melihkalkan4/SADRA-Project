# scripts/plot_results.py
import matplotlib.pyplot as plt
import seaborn as sns
import json, os

def plot_sadra_ranks(rank_config_path, output_path="output/rank_dist.png"):
    if not os.path.exists(rank_config_path):
        print("Hata: rank_config dosyası bulunamadı!")
        return

    with open(rank_config_path, "r") as f:
        data = json.load(f)

    # Verileri temizle ve sırala
    layers = [k.replace("distilbert.transformer.layer.", "L").replace(".weight", "") for k in data.keys()]
    ranks = list(data.values())

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # FFN ve Attention katmanlarını ayırmak için renk paleti
    colors = ["#4C72B0" if "ffn" in l.lower() else "#55A868" for l in layers]
    
    bars = plt.bar(layers, ranks, color=colors, edgecolor='black', alpha=0.8)
    plt.axhline(y=8, color='red', linestyle='--', label='LoRA Static Baseline (r=8)')
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.ylabel("Allocated Rank (r)", fontweight='bold')
    plt.title("SADRA: Dynamic Rank Distribution (Hessian-Aware)", fontsize=14, fontweight='bold')
    plt.legend()
    
    # Bar üzerine değerleri yaz
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, int(yval), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Grafik başarıyla kaydedildi: {output_path}")

if __name__ == "__main__":
    # Eğitim scripti sonucu oluşan json dosyasının yolunu verin
    plot_sadra_ranks("runs/sadra_qnli/seed_42/rank_config_used.json")