"""Behavioral regression tests for the reviewed training revision."""
import copy
import json
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.labels import distance_to_bins, contact_bin_count
from src.data.dataset import PepoTrainDataset
from src.data.collate import collate_fn, get_dataloader
from src.data.fixed_validation import FrozenValidationDataset, UnpaddedShardSampler
from src.model.loss import protein_contact_logit, SurpassLoss
from src.model.res_pair import ResPair
from src.model.ema import EMAWrapper
from src.model.optimizer import get_lr_scheduler
from src.runtime.checkpoint import save_checkpoint, load_checkpoint
from src.runtime.state import capture_rng, preserve_rng
from src.runtime.epoch_inference import inference_due, run_epoch_inference
from src.inference import InferencePairDataset, inference_collate_fn, evaluate_ppi
from src.train import evaluate_epoch, train_epoch

ROOT = Path(__file__).resolve().parents[1]


def config():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name="train")


def model():
    return ResPair(n_layers=2, dim_token=16, dim_pair=8, n_heads=4, plm_in_dim=8,
                   plm_out_dim=8, index_emb_dim=4, triangle_hidden_dim=4,
                   triangle_chunk_size=2, triangle_checkpoint=False,
                   distance_bin_count=22, distance_bin_convention="left_closed", protein_pooling="max")


def fixture(root):
    (root / "embedded").mkdir()
    (root / "labels").mkdir()
    for i in range(4):
        features = {"plm_emb": torch.arange(96).reshape(12, 8).float()/100+i,
                    "residue_type": torch.zeros(12).long(), "residue_index": torch.arange(12),
                    "chain_index": torch.zeros(12).long(), "cb_mask": torch.ones(12).bool()}
        torch.save(features, root / "embedded" / f"p{i}.pt")
        torch.save({"plm_emb": features["plm_emb"]}, root / f"p{i}.pt")
    dist = torch.full((24, 24), 25.)
    dist[1, 13] = dist[13, 1] = 7.9
    dist[10, 22] = dist[22, 10] = 6.0
    with (root / "labels" / "label.pkl").open("wb") as handle:
        pickle.dump({"pairwise_dist": dist, "pairwise_mask": torch.ones(24,24).bool()}, handle)
    rows = []
    for cluster in range(6):
        for row in range(2):
            rows.append({"chain1": f"p{row}", "chain2": f"p{row+2}", "ppi_cluster_id": f"c{cluster}",
                         "source_is_positive": cluster < 3, "label_path": "label.pkl"})
    manifest = root / "pairs.tsv"
    pd.DataFrame(rows).to_csv(manifest, sep="\t", index=False)
    data = PepoTrainDataset(str(root), manifest, crop_size=8, distance_bin_count=22,
                           distance_bin_convention="left_closed")
    (root / "test.fasta").write_text("".join(f">p{i}\n{'A'*12}\n" for i in range(4)))
    (root / "test.tsv").write_text("Protein pairs\tCategory\np0_p2\t1\np1_p3\t0\n")
    return data


class RevisionTest(unittest.TestCase):
    def test_current_defaults_and_compatibility_aliases(self):
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            for current, alias in (("train", "train_revision"), ("inference", "inference_revision")):
                cfg=compose(config_name=current)
                old_name=compose(config_name=alias)
                self.assertEqual(OmegaConf.to_container(cfg,resolve=True),OmegaConf.to_container(old_name,resolve=True))
                self.assertEqual(cfg.model.name,"ResPair")
                self.assertEqual(cfg.model.protein_pooling,"max")
                self.assertEqual(cfg.model.recycle_passes,1)
                self.assertEqual(cfg.data.distance_bin_count,22)
                self.assertEqual(cfg.data.distance_bin_convention,"left_closed")
                self.assertEqual(cfg.runtime.checkpoint_weights,"ema")
            self.assertEqual(compose(config_name="train").loss.protein.weight,0)
            self.assertEqual(compose(config_name="train_respair").model.recycle_passes,2)
            self.assertEqual(compose(config_name="inference_respair").data.distance_bin_count,20)

    def test_exact_bins_and_strict_contact(self):
        distances = torch.tensor([1.99,2.,2.999,3.,7.,7.999,8.,8.001,9.,21.999,22.,23.])
        bins = distance_to_bins(distances,2,22,22,convention="left_closed")
        self.assertEqual(bins.tolist(), [0,1,1,2,6,6,7,7,8,20,21,21])
        self.assertEqual(contact_bin_count(8,2,22,22,convention="left_closed"),7)
        self.assertTrue(torch.equal(bins < 7, distances < 8))
        with self.assertRaises(ValueError):
            contact_bin_count(8,2,22,20,convention="left_closed")

    def test_max_pooling_empty_bag_mask_and_zero_weight(self):
        logits = torch.tensor([[[[1.,0.],[4.,0.]]],[[[1.,0.],[2.,0.]]]],requires_grad=True)
        mask = torch.tensor([[[True,False]],[[False,False]]])
        pooled = protein_contact_logit(logits,mask,1,"max")
        torch.testing.assert_close(pooled,torch.tensor([1.,-80.]))
        pooled.sum().backward()
        self.assertEqual(logits.grad[1].abs().sum().item(),0)
        self.assertEqual(logits.grad[0,0,1].abs().sum().item(),0)
        with self.assertRaises(ValueError):
            protein_contact_logit(logits,mask,1,"unknown")
        with tempfile.TemporaryDirectory() as tmp:
            data=fixture(Path(tmp));batch,labels=collate_fn([data[0],data[4]])
            output=model()(batch)
            output.protein_logit.retain_grad()
            total,_=SurpassLoss(7,{"protein":0,"contact":1,"distance":1,"aux_protein":.1})(output,labels)
            total.backward()
            self.assertEqual(output.protein_logit.grad.abs().sum().item(),0)

    def test_fixed_panel_replay_shards_and_dual_ap(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=fixture(Path(tmp))
            panel=FrozenValidationDataset.build(data,[{"name":"test","weight":1}],6,1,42)
            restored=FrozenValidationDataset(data,json.loads(json.dumps(panel.manifest)))
            first=list(get_dataloader(panel,collate_fn,2,shuffle=False,num_workers=0,pin_memory=False))
            # Worker process start method differs by OS; test actual workers too.
            second=list(get_dataloader(restored,collate_fn,2,shuffle=False,num_workers=1,pin_memory=False))
            for (a,la),(b,lb) in zip(first,second):
                torch.testing.assert_close(a["plm_emb"],b["plm_emb"],rtol=0,atol=0)
                self.assertEqual(la["crop_metadata"],lb["crop_metadata"])
            shards=[list(UnpaddedShardSampler(panel,r,4)) for r in range(4)]
            self.assertEqual(sorted(sum(shards,[])),list(range(6)))
            net=model()
            result=evaluate_epoch(net,first,SurpassLoss(7,{"protein":0,"contact":1,"distance":1,"aux_protein":.1}),torch.device('cpu'),precision='float32',positive_weight=.01)
            self.assertIn('pair_auprc_unweighted',result)
            self.assertIn('contact',result)
            repeated=evaluate_epoch(net,second,SurpassLoss(7,{"protein":0,"contact":1,"distance":1,"aux_protein":.1}),torch.device('cpu'),precision='float32',positive_weight=.01)
            self.assertEqual(result,repeated)
            data.metadata.to_csv(data.cluster_tsv_path,sep='\t',index=False)
            # Corrupt persisted crop data must be rejected before replay.
            broken=copy.deepcopy(panel.manifest);broken['entries'][0]['crop_seed']+=1
            with self.assertRaises(ValueError): FrozenValidationDataset(data,broken)

    def test_ema_complete_resume_and_weight_selection(self):
        torch.manual_seed(3)
        net=model();ema=EMAWrapper(net,.8);ema.register()
        opt=torch.optim.AdamW(net.parameters(),lr=.01)
        sch=get_lr_scheduler(opt,'cosine_annealing',lr=.01,min_lr=.001,max_steps=6,warmup_steps=2)
        def step(n,o,s,e):
            o.zero_grad(); sum(p.square().sum() for p in n.parameters()).backward();o.step();s.step();e.update()
        for _ in range(2): step(net,opt,sch,ema)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pth'
            save_checkpoint(path,net,epoch=2,optimizer=opt,scheduler=sch,ema=ema,training_state={'marker':1})
            other=model();other_ema=EMAWrapper(other,.1)
            other_opt=torch.optim.AdamW(other.parameters(),lr=.01)
            other_sch=get_lr_scheduler(other_opt,'cosine_annealing',lr=.01,min_lr=.001,max_steps=6,warmup_steps=2)
            load_checkpoint(path,other,optimizer=other_opt,scheduler=other_sch,ema=other_ema,load_training_state=True)
            for _ in range(2):
                step(net,opt,sch,ema);step(other,other_opt,other_sch,other_ema)
            for key in net.state_dict():
                torch.testing.assert_close(net.state_dict()[key],other.state_dict()[key],rtol=0,atol=0)
                torch.testing.assert_close(ema.shadow[key],other_ema.shadow[key],rtol=0,atol=0)
            self.assertEqual(sch.state_dict(),other_sch.state_dict())
            saved=torch.load(path,weights_only=False)
            evaluation=model();load_checkpoint(path,evaluation,weights='ema')
            for key,value in evaluation.state_dict().items():
                torch.testing.assert_close(value,saved['ema_state_dict']['shadow'][key])
            changed=model();changed.bin_spec['convention']='legacy'
            with self.assertRaisesRegex(ValueError,'contract'): load_checkpoint(path,changed)
            before={k:v.clone() for k,v in net.state_dict().items()}
            with self.assertRaises(RuntimeError),ema.average_parameters(): raise RuntimeError('test')
            for key in before: torch.testing.assert_close(before[key],net.state_dict()[key])

    def test_scheduler_boundaries_clipping_disabled_partial_accumulation(self):
        net=torch.nn.Linear(1,1)
        opt=torch.optim.SGD(net.parameters(),lr=.01)
        sch=get_lr_scheduler(opt,'cosine_annealing',lr=.01,min_lr=.001,max_steps=6,warmup_steps=2)
        self.assertEqual(opt.param_groups[0]['lr'],.005)
        for _ in range(6):opt.step();sch.step()
        self.assertAlmostEqual(opt.param_groups[0]['lr'],.001)
        with tempfile.TemporaryDirectory() as tmp:
            data=fixture(Path(tmp));batches=[collate_fn([data[0],data[4]])]*3
            net=model();ema=EMAWrapper(net,.9);ema.register()
            opt=torch.optim.SGD(net.parameters(),lr=.001)
            sch=get_lr_scheduler(opt,'constant',lr=.001,max_steps=2)
            with patch('torch.nn.utils.clip_grad_norm_',side_effect=AssertionError('clipping must be disabled')):
                train_epoch(net,batches,opt,sch,SurpassLoss(7,{'protein':0,'contact':1,'distance':1,'aux_protein':.1}),torch.device('cpu'),precision='float32',grad_accum_steps=2,max_grad_norm=0,ema=ema)
            self.assertEqual(ema.num_updates,2)

    def test_periodic_inference_matches_saved_ema_and_preserves_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            args=config()
            net=model();ema=EMAWrapper(net,.9);ema.register()
            args.model=OmegaConf.create({'name':'ResPair','n_layers':2,'dim_token':16,'dim_pair':8,'n_heads':4,'plm_in_dim':8,'plm_out_dim':8,'index_emb_dim':4,'triangle_hidden_dim':4,'triangle_chunk_size':2,'triangle_checkpoint':False,'protein_pooling':'max','recycle_passes':1,'pair_update_every':2})
            args.epoch_inference.config.data.root_dir=str(root)
            args.epoch_inference.config.data.test_set_tsv=str(root/'test.tsv')
            args.epoch_inference.config.data.fasta_path=str(root/'test.fasta')
            args.epoch_inference.config.data.embedding_dir=str(root)
            path=root/'epoch_10.pth';save_checkpoint(path,net,epoch=10,ema=ema)
            before=torch.get_rng_state().clone();live={k:v.clone() for k,v in net.state_dict().items()}
            metrics=run_epoch_inference(args,10,path,root,net,ema,torch.device('cpu'))
            self.assertEqual(metrics['n_scored'],2)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            for k,v in net.state_dict().items():torch.testing.assert_close(v,live[k])
            evaluation=model();load_checkpoint(path,evaluation,weights='ema')
            loader=get_dataloader(InferencePairDataset(root/'test.tsv',root/'test.fasta',root),inference_collate_fn,1,shuffle=False,num_workers=0,pin_memory=False)
            with preserve_rng(): expected=evaluate_ppi(evaluation,loader,torch.device('cpu'),precision='float32',score_threshold=.5)
            actual=pd.read_csv(root/'epoch_inference/epoch_0010/predictions.tsv',sep='\t')
            np.testing.assert_allclose(actual.p_pair,[row['p_pair'] for row in expected],rtol=0,atol=1e-15)
            self.assertEqual(run_epoch_inference(args,10,path,root,net,ema,torch.device('cpu')), {})
            (root/'epoch_inference/epoch_0010/completion.json').write_text('{')
            self.assertEqual(run_epoch_inference(args,10,path,root,net,ema,torch.device('cpu'))['n_scored'],2)
            self.assertEqual([e for e in range(1,31) if inference_due(args,e)],[10,20,30])
            self.assertFalse(inference_due(args,9))
            cfg=OmegaConf.load(root/'epoch_inference/epoch_0010/config.yaml')
            self.assertEqual(cfg.data.distance_bin_count,22)
            self.assertEqual(cfg.runtime.checkpoint_weights,'ema')

    def test_precision_panel_covers_classes_and_lengths(self):
        from types import SimpleNamespace
        from scripts.compare_precision import representative_indices, main
        dataset=SimpleNamespace(items=[{'label': bool(i%2), 'combined_length': 20+i*100} for i in range(20)])
        indices=representative_indices(dataset,8)
        self.assertEqual(len(indices),8)
        self.assertEqual(len(set(indices)),8)
        self.assertIn(18,indices);self.assertIn(19,indices)
        self.assertEqual(sum(dataset.items[i]['label'] for i in indices),4)
        with patch('torch.cuda.is_available',return_value=False),self.assertRaisesRegex(RuntimeError,'CUDA'):
            main.__wrapped__(config())

    def test_main_training_and_epoch_boundary_resume(self):
        from src.train import main
        real_train_epoch = train_epoch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            args=config()
            args.epochs=2;args.checkpoint_interval=1;args.optimizer.warmup_steps=1
            args.data.root_dir=str(root);args.data.num_workers=0;args.data.batch_size=2
            args.data.validation_samples=6;args.data.validation_negative_ratio=1
            args.data.datasets=OmegaConf.create([{'name':'fixture','subdir':'','weight':1,'require_labels':True,'train_tsv':'pairs.tsv','validation_tsv':'pairs.tsv'}])
            args.model=OmegaConf.create({'name':'ResPair','n_layers':2,'dim_token':16,'dim_pair':8,'n_heads':4,'plm_in_dim':8,'plm_out_dim':8,'index_emb_dim':4,'triangle_hidden_dim':4,'triangle_chunk_size':2,'triangle_checkpoint':False,'protein_pooling':'max','recycle_passes':1,'pair_update_every':2})
            args.epoch_inference.config.data.root_dir=str(root)
            args.epoch_inference.config.data.test_set_tsv=str(root/'test.tsv')
            args.epoch_inference.config.data.fasta_path=str(root/'test.fasta')
            args.epoch_inference.config.data.embedding_dir=str(root)
            args.wandb.enabled=False;args.runtime.precision='float32'
            args.logging_dir=str(root/'continuous')
            main.__wrapped__(args)
            full=next((root/'continuous').glob('*/checkpoints/epoch_2.pth'))
            args.logging_dir=str(root/'interrupted')
            calls=0
            def interrupt(*a,**kw):
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError('simulated interruption after epoch 1')
                return real_train_epoch(*a,**kw)
            with patch('src.train.train_epoch',side_effect=interrupt),self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                main.__wrapped__(args)
            args.ckpt_dir=str(next((root/'interrupted').glob('*/checkpoints/epoch_1.pth')))
            # Simulate interruption after checkpoint save but before completed inference.
            pending=Path(args.ckpt_dir).parent.parent/'epoch_inference/epoch_0001/completion.json'
            pending.unlink()
            args.logging_dir=str(root/'resumed')
            main.__wrapped__(args)
            resumed=next((root/'resumed').glob('*/checkpoints/epoch_2.pth'))
            self.assertTrue(pending.is_file())
            a,b=(torch.load(path,weights_only=False) for path in (full,resumed))
            for key in a['model_state_dict']:
                torch.testing.assert_close(a['model_state_dict'][key],b['model_state_dict'][key],rtol=0,atol=0)
                torch.testing.assert_close(a['ema_state_dict']['shadow'][key],b['ema_state_dict']['shadow'][key],rtol=0,atol=0)
            self.assertEqual(a['training_state']['validation_panel']['identity'],b['training_state']['validation_panel']['identity'])
            self.assertEqual(a['scheduler_state_dict'],b['scheduler_state_dict'])
            self.assertTrue((resumed.parent.parent/'epoch_inference/epoch_0002/completion.json').is_file())

if __name__=='__main__':unittest.main()
