"""Matched CUDA bf16/float32 inference and short-training diagnostics.

Run with: python -m scripts.compare_precision [training overrides]
This is separate from training and never changes pilot checkpoints.
"""
import json
import math
import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from src.data.collate import get_dataloader, collate_fn
from src.inference import InferencePairDataset, inference_collate_fn, evaluate_ppi
from src.model.loss import SurpassLoss
from src.model.optimizer import get_optimizer, get_lr_scheduler
from src.runtime.checkpoint import load_checkpoint
from src.runtime.config import create_model, cfg_get
from src.runtime.device import autocast_context, to_device
from src.runtime.metrics import metrics_from_rows
from src.utils.ddp_utils import seed_everything
from src.train import _build_datasets


def representative_indices(dataset, budget):
    """Spread a bounded panel across class and length, including long pairs."""
    if budget < 1:
        raise ValueError("precision_check.samples must be positive")
    pools = [[i for i, item in enumerate(dataset.items) if item['label'] == label] for label in (False, True)]
    result = []
    for pool in pools:
        pool.sort(key=lambda i: dataset.items[i]['combined_length'])
        take = min(len(pool), max(1, budget // 2))
        if take:
            result.extend(pool[round(j*(len(pool)-1)/max(1,take-1))] for j in range(take))
    selected = set(result)
    remaining = [i for i in range(len(dataset.items)) if i not in selected]
    return sorted((result + remaining[:max(0,budget-len(result))])[:budget])


@hydra.main(version_base='1.3', config_path='../configs', config_name='train')
def main(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Precision comparison requires a CUDA GPU supporting bf16')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Run this diagnostic on one GPU, without a multi-rank launcher')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    seed_everything(int(args.seed), bool(args.deterministic))
    if int(args.optimizer.grad_accum_steps) != 1:
        raise ValueError('This matched diagnostic currently requires grad_accum_steps=1')
    output = Path(args.precision_check.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(args, output / 'config.yaml', resolve=True)
    cfg = args.epoch_inference.config
    dataset = InferencePairDataset(cfg.data.test_set_tsv, cfg.data.fasta_path, cfg.data.embedding_dir,
                                  maximum_length=int(cfg.data.maximum_length), embedding_cache_mb=0)
    indices = representative_indices(dataset, int(args.precision_check.samples))
    if not indices:
        raise ValueError('No supported inference examples for precision comparison')
    loader = get_dataloader(torch.utils.data.Subset(dataset, indices), inference_collate_fn, 1,
                            shuffle=False, num_workers=0, pin_memory=False)
    net = create_model(args).to(device)
    if args.ckpt_dir:
        load_checkpoint(args.ckpt_dir, net, map_location='cpu', weights=str(args.runtime.checkpoint_weights))
    report = {'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0), 'tf32': False,
              'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              'attention_backends': {'flash_enabled': torch.backends.cuda.flash_sdp_enabled(),
                                     'math_enabled': torch.backends.cuda.math_sdp_enabled(),
                                     'memory_efficient_enabled': torch.backends.cuda.mem_efficient_sdp_enabled()},
              'checkpoint': str(args.ckpt_dir) if args.ckpt_dir else 'initialized_model',
              'panel_indices': indices, 'inference': {}, 'training': {}}
    results = {}
    for precision in ('float32','bfloat16'):
        results[precision] = evaluate_ppi(net, loader, device, precision=precision,
                                         score_threshold=float(cfg.data.ppi_score_threshold))
        report['inference'][precision] = {
            'weighted': metrics_from_rows(results[precision],'p_pair',positive_weight=.01),
            'unweighted': metrics_from_rows(results[precision],'p_pair',positive_weight=1),
            'distinct_scores': len({r['p_pair'] for r in results[precision]})}
        (output / f'inference_{precision}.json').write_text(json.dumps(results[precision],indent=2))
    error = torch.tensor([abs(a['p_pair']-b['p_pair']) for a,b in zip(results['float32'],results['bfloat16'])])
    report['inference']['score_error'] = {'mean_abs': error.mean().item(), 'max_abs': error.max().item()}
    del net
    # Bound host memory: prepare a few fixed batches, replay them in both runs.
    seed_everything(int(args.seed), bool(args.deterministic))
    _, train_data, _, _, _ = _build_datasets(args)
    train_loader = get_dataloader(train_data, collate_fn, int(args.data.batch_size),
                                  shuffle=False, num_workers=0, pin_memory=False)
    batches = []
    for batch in train_loader:
        batches.append(batch)
        if len(batches) >= int(args.precision_check.train_batches): break
    if not batches:
        raise ValueError('No training batches for precision comparison')
    for precision in ('float32','bfloat16'):
        # Reinitialization with the same seed gives identical starting weights.
        seed_everything(int(args.seed), bool(args.deterministic))
        net = create_model(args).to(device).train()
        optimizer = get_optimizer(net, lr=float(args.optimizer.lr), weight_decay=float(args.optimizer.weight_decay),
                                  betas=(float(args.optimizer.beta1),float(args.optimizer.beta2)), use_adamw=True)
        scheduler = get_lr_scheduler(optimizer, lr_scheduler=str(args.optimizer.lr_scheduler),
            lr=float(args.optimizer.lr), min_lr=float(args.optimizer.min_lr),
            max_steps=int(args.epochs)*len(train_loader), warmup_steps=int(args.optimizer.warmup_steps),
            decay_every_n_steps=int(args.optimizer.decay_every_n_steps), decay_factor=float(args.optimizer.decay_factor))
        loss = SurpassLoss(net.contact_bins, {name:float(args.loss[name].weight) for name in ('protein','contact','distance','aux_protein')},
                           focal_gamma=float(args.loss.contact.gamma), contact_positive_weight=float(args.loss.contact.positive_weight),
                           negatives_per_positive=float(args.loss.contact.negatives_per_positive))
        history=[]
        for step in range(int(args.precision_check.training_updates)):
            inputs, labels = (to_device(v,device) for v in batches[step % len(batches)])
            optimizer.zero_grad(set_to_none=True)
            lr = float(optimizer.param_groups[0]['lr'])
            with autocast_context(device,precision):
                total, terms = loss(net(inputs),labels)
            finite=bool(torch.isfinite(total))
            history.append({**{k:float(v) for k,v in terms.items()},'finite':finite,'lr':lr})
            if not finite:break
            total.backward()
            if not all(bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.grad is not None):
                history[-1]['finite_gradients']=False
                break
            if float(args.optimizer.max_grad_norm)>0:
                torch.nn.utils.clip_grad_norm_(net.parameters(),float(args.optimizer.max_grad_norm))
            optimizer.step()
            scheduler.step()
        report['training'][precision]=history
        del loss, optimizer, scheduler, net
    (output/'comparison.json').write_text(json.dumps(report,indent=2))
    print(f'Precision comparison written to {output}')


if __name__=='__main__':main()
