# src/sadra/core.py
from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import Dict, Any

def estimate_hessian_trace(
    model: nn.Module, 
    data_loader: Any, 
    device: str = "cuda", 
    num_batches: int = 2, 
    num_vectors: int = 10
) -> Dict[str, float]:
    """
    Hutchinson's Trace Estimator kullanarak model katmanlarının Hessian Trace (duyarlılık) skorlarını hesaplar.
    
    Teorik Temel: SADRA, katman hassasiyetini v^T * H * v beklenen değeriyle ölçer.
    """
    print(f"\n[SADRA] Hessian Analizi Başlatılıyor: {num_batches} batch ve katman başına {num_vectors} vektör.")
    
    model.eval()
    model.zero_grad()
    
    layer_scores: Dict[str, float] = {}
    
    # KRİTİK: Flash Attention 2. türevleri (Hessian) desteklemez.
    # sdpa_kernel(SDPBackend.MATH) kullanarak güvenli hesaplama moduna geçiyoruz.
    with sdpa_kernel(SDPBackend.MATH):
        
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
                
            # Veriyi cihaza taşı (Hugging Face / PyTorch formatı uyumlu)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 1. Forward Pass & Loss
            outputs = model(**batch)
            loss = outputs.loss
            
            # 2. First Backward: Hessian (türevin türevi) için grafiği koru
            loss.backward(create_graph=True)
            
            for name, param in model.named_parameters():
                # SADRA: Sadece ağırlık matrislerine (weight) odaklan
                if "weight" not in name or not param.requires_grad:
                    continue
                
                if param.grad is not None:
                    grad = param.grad
                    trace_sum = 0.0
                    
                    # Hutchinson Döngüsü
                    for _ in range(num_vectors):
                        # Rademacher Dağılımı (±1): En verimli trace estimator
                        v = torch.randint_like(param, high=2) * 2 - 1
                        v = v.float().to(device)
                        
                        # grad_v_prod = <∇L, v> (İç çarpım)
                        grad_v_prod = torch.sum(grad * v)
                        
                        # 3. Second Backward: Hv = ∇(<∇L, v>)
                        # retain_graph=True: Aynı batch içinde birden fazla vektör deneneceği için.
                        hvp = torch.autograd.grad(
                            grad_v_prod, 
                            param, 
                            retain_graph=True,
                            create_graph=False
                        )[0]
                        
                        # Trace Estimate: v^T * H * v = v^T * hvp
                        trace_sum += torch.sum(v * hvp).item()
                    
                    if name not in layer_scores:
                        layer_scores[name] = 0.0
                    layer_scores[name] += trace_sum / num_vectors

            # Bellek sızıntısını önlemek için gradyanları temizle
            model.zero_grad(set_to_none=True)

    # Batch ortalaması ile final skorlarını hesapla
    final_scores = {name: score / num_batches for name, score in layer_scores.items()}
    print(f"[SADRA] Analiz tamamlandı. {len(final_scores)} katman değerlendirildi.")
    
    return final_scores