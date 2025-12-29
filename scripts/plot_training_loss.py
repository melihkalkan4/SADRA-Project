# scripts/plot_training_loss.py
import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_loss_curves(root="runs/sadra_qnli"):
    # Log dosyalarını bul
    log_files = glob.glob(os.path.join(root, "seed_*", "checkpoint-*", "trainer_state.json"))
    # Eğer checkpoint içinde bulamazsa ana klasöre bak (Trainer bazen oraya yazar)
    if not log_files:
         log_files = glob.glob(os.path.join(root, "seed_*", "trainer_state.json"))

    data = []
    
    for log_path in log_files:
        seed = os.path.dirname(log_path).split("_")[-1]
        if "checkpoint" in seed: # Path düzeltmesi
            seed = os.path.dirname(os.path.dirname(log_path)).split("_")[-1]
            
        with open(log_path, "r") as f:
            state = json.load(f)
            
        history = state.get("log_history", [])
        for entry in history:
            if "loss" in entry and "epoch" in entry:
                data.append({
                    "Epoch": entry["epoch"],
                    "Training Loss": entry["loss"],
                    "Seed": seed
                })

    if not data:
        print("Hata: Hiçbir eğitim logu (trainer_state.json) bulunamadı.")
        return

    df = pd.DataFrame(data)

    # Grafik Ayarları
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Çizgi Grafik (Ortalama ve Standart Sapma gölgeli)
    sns.lineplot(data=df, x="Epoch", y="Training Loss", errorbar="sd", linewidth=2.5, color="#1f77b4")
    
    plt.title("SADRA: Training Convergence (Mean of 3 Seeds)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.tight_layout()
    
    out_path = "output/sadra_loss_curve.png"
    os.makedirs("output", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Loss grafiği oluşturuldu: {out_path}")

if __name__ == "__main__":
    plot_loss_curves()