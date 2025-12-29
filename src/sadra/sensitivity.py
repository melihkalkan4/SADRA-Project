# sadra/sensitivity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn


@dataclass
class SensitivityConfig:
    n_batches: int = 8              # how many mini-batches to probe
    hvp_samples: int = 4            # Hutchinson samples per batch
    max_modules: Optional[int] = None  # optionally limit to top modules by size later
    use_fp16: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _collect_candidate_modules(model: nn.Module, module_types=(nn.Linear,)) -> Dict[str, nn.Module]:
    """
    Return {module_name: module} candidates for LoRA/Hessian scoring.
    Default: nn.Linear only (good for Transformer blocks).
    """
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, module_types):
            out[name] = module
    return out


def _module_param(module: nn.Module) -> torch.Tensor:
    # For nn.Linear, LoRA typically targets weight. Use weight if exists.
    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        return module.weight
    # fallback: first parameter
    for p in module.parameters(recurse=False):
        return p
    raise ValueError("Module has no parameters.")


@torch.no_grad()
def _estimate_module_sizes(modules: Dict[str, nn.Module]) -> Dict[str, int]:
    sizes = {}
    for name, m in modules.items():
        p = _module_param(m)
        sizes[name] = p.numel()
    return sizes


def compute_hessian_sensitivity(
    model: nn.Module,
    loss_fn,
    dataloader,
    *,
    cfg: SensitivityConfig = SensitivityConfig(),
    module_types=(nn.Linear,),
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Approx sensitivity score per module:
      score(m) ~ E_z [ || H_m * z ||^2 ]   (Hessian-vector probes)
    computed via autograd Hessian-vector products.

    Notes:
    - This is not full Hessian, it's an efficient proxy.
    - Works best when you probe a few batches from the training set.
    """

    model.eval()
    model.to(cfg.device)

    candidates = _collect_candidate_modules(model, module_types=module_types)
    if len(candidates) == 0:
        raise ValueError("No candidate modules found. Check module_types.")

    # Optionally restrict candidate set (useful for huge models)
    if cfg.max_modules is not None and len(candidates) > cfg.max_modules:
        sizes = _estimate_module_sizes(candidates)
        # keep biggest modules
        top = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[: cfg.max_modules]
        keep = set([k for k, _ in top])
        candidates = {k: v for k, v in candidates.items() if k in keep}

    # Make a list of params for HVP
    names = list(candidates.keys())
    params = [ _module_param(candidates[n]) for n in names ]

    # We need grads for these params
    for p in params:
        p.requires_grad_(True)

    scores = {n: 0.0 for n in names}
    total_probes = 0

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    if verbose:
        print(f"[SADRA] Hessian sensitivity: modules={len(names)}, n_batches={cfg.n_batches}, hvp_samples={cfg.hvp_samples}")

    batch_iter = iter(dataloader)
    for b in range(cfg.n_batches):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(dataloader)
            batch = next(batch_iter)

        # Move batch to device
        batch = {k: (v.to(cfg.device) if hasattr(v, "to") else v) for k, v in batch.items()}

        # Forward + loss
        with torch.cuda.amp.autocast(enabled=cfg.use_fp16):
            out = model(**batch)
            loss = out.loss if hasattr(out, "loss") else loss_fn(out, batch["labels"])

        # First-order grads
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)

        # Hutchinson probes
        for _ in range(cfg.hvp_samples):
            # random Rademacher vectors for each param
            zs = []
            for p in params:
                z = torch.empty_like(p).bernoulli_(0.5).mul_(2).add_(-1)  # ±1
                zs.append(z)

            # <grad, z>
            inner = 0.0
            for g, z in zip(grads, zs):
                if g is None:
                    continue
                inner = inner + (g * z).sum()

            # HVP = d/dp <grad, z>  => gives (H * z)
            hvps = torch.autograd.grad(inner, params, retain_graph=True, allow_unused=True)

            # accumulate ||H z||^2 per module
            for n, hv in zip(names, hvps):
                if hv is None:
                    continue
                scores[n] += float((hv.detach() ** 2).mean().cpu().item())

            total_probes += 1

        # cleanup graph
        del loss, out, grads, inner, hvps

    # average
    for n in scores:
        scores[n] /= max(total_probes, 1)

    if verbose:
        # show top 10
        top10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        print("[SADRA] Top-10 sensitive modules:")
        for k, v in top10:
            print(f"  {k:60s}  {v:.6e}")

    return scores
