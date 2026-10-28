"""Real two-rank CPU smoke test for the revised training/evaluation lifecycle."""
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
import unittest

import pandas as pd
import torch
from omegaconf import OmegaConf
from tests.test_revision import fixture, config, ROOT


class DistributedRevisionTest(unittest.TestCase):
    def test_two_rank_training_validation_and_full_inference(self):
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
            args.wandb.enabled=False;args.runtime.precision='float32';args.logging_dir=str(root/'logs')
            OmegaConf.save(args,root/'smoke.yaml',resolve=True)
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                 'GLOO_SOCKET_IFNAME':'lo0' if sys.platform=='darwin' else 'lo'}
            with socket.socket() as listener:
                listener.bind(('127.0.0.1',0))
                port=listener.getsockname()[1]
            try:
                result=subprocess.run([sys.executable,'-m','torch.distributed.run','--master_addr=127.0.0.1',f'--master_port={port}','--nproc_per_node=2',
                '-m','src.train','--config-dir',str(root),'--config-name','smoke'],cwd=ROOT,env=env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=45)
            except subprocess.TimeoutExpired as exc:
                self.fail(f'Distributed smoke timeout: {exc.stdout!r}')
            self.assertEqual(result.returncode,0,result.stdout[-12000:])
            checkpoint=next((root/'logs').glob('*/checkpoints/epoch_2.pth'))
            payload=torch.load(checkpoint,weights_only=False)
            self.assertEqual(payload['ema_state_dict']['num_updates'],4)
            self.assertEqual(len(payload['training_state']['rng_states']),2)
            run=checkpoint.parent.parent
            metrics=pd.read_csv(run/'metrics.csv')
            self.assertEqual(metrics.validation_samples.tolist(),[6,6])
            predictions=pd.read_csv(run/'epoch_inference/epoch_0002/predictions.tsv',sep='\t')
            self.assertEqual(len(predictions),2)
            self.assertEqual(predictions['Protein pairs'].nunique(),2)
            self.assertEqual(json.loads((run/'epoch_inference/epoch_0002/completion.json').read_text())['status'],'complete')

if __name__=='__main__':unittest.main()
