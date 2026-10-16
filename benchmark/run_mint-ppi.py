import torch
from mint.helpers.extract import load_config, CSVDataset, CollateFn, MINTWrapper
from mint.helpers.predict import SimpleMLP

cfg = load_config("data/esm2_t33_650M_UR50D.json") # model config
device = 'cuda:0' # GPU device
checkpoint_path = 'mint.ckpt' # Where you stored the model checkpoint
mlp_checkpoint_path = 'bernett_mlp.pth' # Where you stored the Bernett MLP checkpoint

dataset = CSVDataset('data/protein_sequences.csv', 'Protein_Sequence_1', 'Protein_Sequence_2')
loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=CollateFn(512), shuffle=False) 

wrapper = MINTWrapper(cfg, checkpoint_path, sep_chains=True, device=device)

# Generate embeddings 
chains, chain_ids = next(iter(loader)) 
chains = chains.to(device)
chain_ids = chain_ids.to(device)
embeddings = wrapper(chains, chain_ids) # Should be of shape (2, 2560)

# Predict using trained MLP
model = SimpleMLP() 
mlp_checkpoint = torch.load(mlp_checkpoint_path)
model.load_state_dict(mlp_checkpoint)
model.eval()
model.to(device)

predictions = torch.sigmoid(model(embeddings)) # Should be of shape (2, 1)
print(predictions) # Probability of interaction (0 is no, 1 is yes)