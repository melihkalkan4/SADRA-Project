import torch
import torch.nn as nn
from torch.backends.cuda import sdp_kernel

def estimate_hessian_trace(model, data_loader, device="cuda", num_batches=2, num_vectors=10):
    """
    Hutchinson's Trace Estimator kullanarak modelin katmanlarının
    duyarlılık (Hessian Trace) skorlarını hesaplar.
    
    Args:
        model: Fine-tune edilecek model
        data_loader: Birkaç örnek veri içeren loader
        device: 'cuda' veya 'cpu'
        num_batches: Kaç batch veri kullanılacağı
        num_vectors: Hutchinson için kaç rastgele vektör üretileceği
    
    Returns:
        sensitivity_scores: {layer_name: trace_value} sözlüğü
    """
    print(f"SADRA: Hessian Trace tahmini başlatılıyor... (Batches: {num_batches}, Vectors: {num_vectors})")
    
    model.eval()
    model.zero_grad()
    
    # Skorları tutacak sözlük
    layer_scores = {}
    
    # Flash Attention'ı devre dışı bırakıyoruz. 
    # Çünkü Flash Attention 2. türevleri (Hessian) desteklemez.
    # enable_math=True diyerek klasik ve güvenli hesaplamayı açıyoruz.
    with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        
        # Veri üzerinde döngü
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
                
            # Veriyi Cihaza Taşı
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward Pass (Loss Hesapla)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # 1. Gradyanları Hesapla (First Backward)
            # create_graph=True Hessian için şarttır.
            loss.backward(create_graph=True)
            
            # Her bir parametre grubu (katman) için HVP hesapla
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Sadece ağırlık matrislerine (weight) odaklanalım, bias'ları geçelim (daha temiz sonuç için)
                if "weight" not in name:
                    continue
                
                if param.grad is not None:
                    grad = param.grad
                    
                    # Hutchinson Döngüsü (Random Vektörler)
                    trace_sum = 0.0
                    for _ in range(num_vectors):
                        # Rademacher Dağılımı (+1 veya -1)
                        v = torch.randint_like(param, high=2) * 2 - 1
                        v = v.float().to(device)
                        
                        # Gradyan ile vektörün çarpımı (skaler)
                        grad_v_prod = torch.sum(grad * v)
                        
                        # Bu çarpımın parametreye göre türevi = Hv
                        # retain_graph=True diyerek grafiği koruyoruz
                        hvp = torch.autograd.grad(
                            grad_v_prod, 
                            param, 
                            retain_graph=True,
                            create_graph=False
                        )[0]
                        
                        # Trace tahmini: v^T * H * v -> v^T * hvp
                        trace_sum += torch.sum(v * hvp).item()
                    
                    # Ortalamayı kaydet
                    current_score = trace_sum / num_vectors
                    if name not in layer_scores:
                        layer_scores[name] = 0.0
                    layer_scores[name] += current_score

            # Gradyanları temizle (Hafıza sızıntısını önlemek için grad=None yapıyoruz)
            model.zero_grad(set_to_none=True)

    # Batch ortalamasını al
    for name in layer_scores:
        layer_scores[name] /= num_batches
        
    print(f"SADRA: Hesaplama tamamlandı. {len(layer_scores)} katman analiz edildi.")
    return layer_scores