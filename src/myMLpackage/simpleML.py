# always import comet_ml before torch
from comet_ml import start
from comet_ml.integration.pytorch import log_model

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def make_mlp(in_features, width, nlayer, for_inference=False, binary=True):
    """Build a customizable MLP with optional sigmoid for binary inference."""
    layers = []
    for _ in range(nlayer):
        layers.append(nn.Linear(in_features, width))
        layers.append(nn.ReLU())
        in_features = width
    
    if binary:
        layers.append(nn.Linear(in_features, 1))
        if for_inference:
            layers.append(nn.Sigmoid())
    else:
        if for_inference:
            layers.append(nn.Softmax(dim=-1))  # Fix: added dim for Softmax
    
    return nn.Sequential(*layers)

class Classifier(nn.Module):
    """A simple MLP-based classifier wrapper."""
    def __init__(self, for_inference=False, width=32, nlayer=4, in_features=2, binary=True):
        super().__init__()
        self.fc1 = make_mlp(in_features, width, nlayer, for_inference, binary)

    def forward(self, x):
        return self.fc1(x)

class GaussianDataset(Dataset):
    """2D Gaussian blob dataset with balanced class labels."""
    def __init__(self, n_samples_per_class=10000, mean0=[-2, -2], mean1=[2, 2], cov=[[1, 0], [0, 1]]):
        super().__init__()
        
        self.data0 = np.random.multivariate_normal(mean0, cov, int(n_samples_per_class))
        self.labels0 = np.zeros(int(n_samples_per_class))
        
        self.data1 = np.random.multivariate_normal(mean1, cov, int(n_samples_per_class))
        self.labels1 = np.ones(int(n_samples_per_class))

        self.data = np.vstack((self.data0, self.data1)).astype(np.float32)
        self.labels = np.hstack((self.labels0, self.labels1)).astype(np.int64)

        # Shuffle the combined dataset
        perm = np.random.permutation(len(self.labels))
        self.data = self.data[perm]
        self.labels = self.labels[perm]

        # Class ratio used for loss weighting (e.g., positive/negative imbalance)
        self.ratio = (len(self.labels) - np.sum(self.labels)) / np.sum(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)

def eval_step(model, loss_fn, train_loader, val_loader, device, min_len_train, min_len_val):
    """Evaluate model on training and validation data using distributed averaging."""
    model.eval()
    train_loss_total = torch.tensor(0.0, device=device)
    val_loss_total = torch.tensor(0.0, device=device)
    train_batches = torch.tensor(0, device=device)
    val_batches = torch.tensor(0, device=device)

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= min_len_val: break
            x, y = x.to(device).float(), y.to(device).float()
            val_loss_total += loss_fn(model(x), y.view(-1, 1))
            val_batches += 1

        for i, (x, y) in enumerate(train_loader):
            if i >= min_len_train: break
            x, y = x.to(device).float(), y.to(device).float()
            train_loss_total += loss_fn(model(x), y.view(-1, 1))
            train_batches += 1

    # Aggregate results across all GPUs
    for t in [train_loss_total, train_batches, val_loss_total, val_batches]:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    train_loss = (train_loss_total / train_batches).item() if train_batches.item() > 0 else 0.0
    val_loss = (val_loss_total / val_batches).item() if val_batches.item() > 0 else 0.0

    return {'train_loss': train_loss, 'validation_loss': val_loss}

def train_loop(args):
    # --- DDP: Setup distributed training environment ---
    rank = int(os.environ["RANK"]) # in [0,WORLD_SIZE]
    world_size = int(os.environ["WORLD_SIZE"]) # N_nodes * N_GPUS_per_node
    local_rank = int(os.environ["LOCAL_RANK"]) # in [0,N_GPUS_per_node], used for setting the right GPU
    
    dist.init_process_group(backend='nccl', init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # --- Experiment setup for Comet logging (only on rank 0) ---
    hyper_params = {
        "learning_rate": args.lr,
        "epochs": args.ep,
        "batch_size": args.bs,
        "num_workers": args.num_workers,
    }

    persistent_workers = args.num_workers > 0
    experiment_name = f'lr{args.lr}_bs{args.bs}'
    best_model_params_path = f'{args.out}/models/{experiment_name}.pt'

    if rank == 0:
        experiment = start(
            api_key=args.api_key,
            project_name=args.project_name,
            workspace=args.ws,
        )
        experiment.set_name(experiment_name)
        experiment.log_parameter("exp_key", experiment.get_key())
        experiment.log_parameters(hyper_params)
        print(experiment.get_key())
    else:
        experiment = None

    # --- Model and optimizer setup ---
    model = Classifier(for_inference=False, width=32, nlayer=4, in_features=2, binary=True)
    if rank == 0:
        print(model)
        print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.Adam(model.parameters(), args.lr)

    # --- Dataset creation and dataloader setup ---

    # --- Dataloaders with Distributed Samplers ---
    # ---------------------------------------------
    train_dataset = GaussianDataset(n_samples_per_class=10000, mean0=[-2, -2], mean1=[2, 2], cov=[[1, 0], [0, 1]])
    val_dataset = GaussianDataset(n_samples_per_class=1000, mean0=[-2, -2], mean1=[2, 2], cov=[[1, 0], [0, 1]])

    # -- this will create N_GPUS_per_node copies so unless you can fit N_GPUS_per_node * data_size on the node (500GB max on MPCDF) 
    # -- you should send different chunks (shards) of data to different nodes/gpus explicitly (see commented code below)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, sampler=train_sampler, num_workers=args.num_workers, persistent_workers=persistent_workers)
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, sampler=val_sampler, num_workers=args.num_workers, persistent_workers=persistent_workers)
    # ---------------------------------------------

    # --- Dataloaders with explicit sharding ---    
    # ---------------------------------------------
    # -- each rank will load a different chunk of data, so in real applications make sure this is shuffled before creating shards
    #train_dataset = GaussianDataset(n_samples_per_class=10000 // world_size)
    #val_dataset = GaussianDataset(n_samples_per_class=1000 // world_size)

    #train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, persistent_workers=persistent_workers)
    #val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, persistent_workers=persistent_workers)
    # ---------------------------------------------

    # --- Loss function setup with synchronized class weight ---
    # Synchronize a loss weight (e.g., class ratio) across all processes in a distributed setting. 
    # Only rank 0 computes the actual value; others receive it via broadcast.
    if rank == 0:
        ratio_tensor = torch.tensor([train_dataset.ratio], dtype=torch.float32).to(device)
    else:
        ratio_tensor = torch.zeros(1, dtype=torch.float32).to(device)
    dist.broadcast(ratio_tensor, src=0)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=ratio_tensor)  

    # --- Determine min batch count seen across all ranks (all ranks should see the same amount of batches to avoid sync issues) ---
    min_len_train = torch.tensor(len(train_loader), device=device)
    min_len_val = torch.tensor(len(val_loader), device=device)
    dist.all_reduce(min_len_train, op=dist.ReduceOp.MIN)
    dist.all_reduce(min_len_val, op=dist.ReduceOp.MIN)

    # --- training loop ---
    best_val_loss = float('inf')
    for epoch in range (0,args.ep):
        if rank==0: print(f'epoch: {epoch+1}') 
        model.train()
        for i, (x, y) in enumerate( train_loader ):
            if i >= min_len_train: break
            x = x.to(device).float()
            y = y.to(device).float()
            
            preds = model(x) 
            loss = loss_fn(preds,y.view(-1, 1))
            loss.backward() # here DDP makes all_reduce() sync kick in
            opt.step()
            opt.zero_grad()

        # --- End of Epoch Eval ---    
        eval_result = eval_step(model, loss_fn, train_loader, val_loader, device, min_len_train, min_len_val)
        val_loss = eval_result['validation_loss']     
        if rank == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)
            experiment.log_metrics({
                    "train_loss": eval_result['train_loss'],
                    "val_loss": val_loss
                }, step=epoch)
    # --- Cleanup ---
    dist.destroy_process_group()