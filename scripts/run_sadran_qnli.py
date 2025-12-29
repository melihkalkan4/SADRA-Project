# scripts/run_sadran_qnli.py
from __future__ import annotations
import os, json, argparse, torch, evaluate, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
)
from peft import TaskType

# RTX 3050/3060 (Ampere) için Donanımsal Optimizasyonlar
if torch.cuda.is_available():
    # TensorFloat-32'yi aktif ederek matris çarpımlarını hızlandırır ve kararlı kılar
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# SADRA Çekirdek Modülleri
from sadra.engine import apply_sadra_to_model
from sadra.core import estimate_hessian_trace
from sadra.rank_allocator import allocate_ranks, RankAllocConfig

def get_linear_shapes(model: nn.Module) -> Dict[str, Tuple[int, int]]:
    shapes = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "weight"):
            out_dim, in_dim = module.weight.shape
            shapes[name] = (in_dim, out_dim)
    return shapes

def main():
    parser = argparse.ArgumentParser(description="SADRA QNLI Training Script (Ampere Optimized)")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--out_dir", type=str, default="runs/sadra_qnli")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 13, 123])
    parser.add_argument("--auto_rank", action="store_true")
    parser.add_argument("--lora_budget", type=int, default=600000)
    parser.add_argument("--rank_max", type=int, default=32)
    parser.add_argument("--rank_min", type=int, default=4) # RTX 3050 için min rank 4 daha güvenlidir
    parser.add_argument("--rank_power", type=float, default=1.0)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=8) # VRAM güvenliği için 8 veya 16 önerilir
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--alpha_ratio", type=float, default=2.0)
    parser.add_argument("--bf16", action="store_true", help="RTX 30-serisi için en stabil mod")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in args.seeds:
        print(f"\n>>> İşlem Başlatılıyor: Seed {seed}")
        set_seed(seed)
        run_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        # 1. Veri Hazırlığı ve Sütun Temizliği
        ds = load_dataset("glue", "qnli")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        def tokenize_fn(ex): 
            return tokenizer(ex["question"], ex["sentence"], truncation=True, max_length=256, padding="max_length")
        
        tokenized = ds.map(tokenize_fn, batched=True)
        # ValueError: too many dimensions 'str' hatasını önlemek için metin sütunlarını kaldır
        model_inputs = ["input_ids", "attention_mask", "label", "token_type_ids"]
        cols_to_remove = [c for c in tokenized["train"].column_names if c not in model_inputs]
        tokenized = tokenized.remove_columns(cols_to_remove)
        tokenized.set_format("torch")
        
        collator = DataCollatorWithPadding(tokenizer)

        # 2. Model ve SADRA Analiz
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

        if args.auto_rank:
            print("[SADRA] Hessian Analizi Başlatılıyor...")
            probe_ds = tokenized["train"].shuffle(seed=seed).select(range(128))
            probe_loader = DataLoader(probe_ds, batch_size=4, collate_fn=collator)
            
            # core.py kullanımı
            sensitivity_scores = estimate_hessian_trace(model, probe_loader, device=device)
            
            # rank_allocator.py kullanımı
            rank_config = allocate_ranks(sensitivity_scores, get_linear_shapes(model), 
                                         RankAllocConfig(lora_budget=args.lora_budget, 
                                                         rank_min=args.rank_min, 
                                                         rank_max=args.rank_max, 
                                                         power=args.rank_power))
            
            # Classifier Çakışma Çözümü (TypeError: modules_to_save hatası için)
            rank_config = {k: v for k, v in rank_config.items() 
                           if "classifier" not in k.lower() and "pre_classifier" not in k.lower()}
            
            with open(os.path.join(run_dir, "rank_config_used.json"), "w") as f:
                json.dump(rank_config, f, indent=2)
        else:
            rank_config = {n: 8 for n in get_linear_shapes(model).keys() if "classifier" not in n.lower()}

        # 3. SADRA Engine Uygulama
        model = apply_sadra_to_model(model, rank_config, alpha_ratio=args.alpha_ratio)

        # 4. Eğitim Ayarları (BF16 Desteği ile)
        train_args = TrainingArguments(
            output_dir=run_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            eval_strategy="epoch", # Yeni transformers versiyonu uyumu
            save_strategy="no",
            logging_steps=100,
            bf16=args.bf16, # RTX 3050 için önerilir
            fp16=args.fp16 if not args.bf16 else False,
            report_to="none"
        )

        metric = evaluate.load("accuracy")
        def compute_metrics(p): return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

        trainer = Trainer(
            model=model, args=train_args, 
            train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
            data_collator=collator, compute_metrics=compute_metrics
        )

        trainer.train()
        eval_results = trainer.evaluate()
        with open(os.path.join(run_dir, "eval_metrics.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"[SEED {seed}] Accuracy: {eval_results['eval_accuracy']:.4f}")

if __name__ == "__main__":
    main()