from matplotlib import pyplot as plt
import h5py
import torch
import torch.nn.functional as F
import numpy as np
import einops
import tqdm
import os

torch.set_float32_matmul_precision('high')

BATCH_SIZE = 4 #how many examples we take simultaneously for the model to get updated
EMBEDDING_DIM = 512 #how fat is the model
N_HEADS = 8
ATTN_DROPOUT = 0.2
TEMPORAL_DIM = 4 #how many bins we give to the model for prediction, we have 4 bins for each neuoron and we prodict
#how many firings will be in the next bin 
EMA = 0.99

BINS = 64
# BINS = 10
BINS_PER_MINUTE = 60000 // BINS
MAX_SPIKES_PER_BIN = 7

assert EMBEDDING_DIM % N_HEADS == 0

data = np.array(h5py.File('231023_22426_112_t_spk_mat_sorted.mat')['t_spk_mat'], dtype=np.int32)
# data.shape (970, 1800023)

data = data[:, :data.shape[1] - (data.shape[1] % BINS)]
#data.shape[1] % BINS This tells us how many extra columns you have that do not fit into an exact multiple of BINS. (gives 23)
#data[:, :new_column_size] uses slicing to select all rows (:) and the first new_column_size columns of the array, where new_column_size is the result of the subtraction in step 3.

data = data.reshape((data.shape[0], data.shape[1] // BINS, BINS))
# output: (970, 28125, 64) This indicates that for each of the 970 rows, the data was organized into 28125 groups (or "bins"), each containing 64 elements.
n_spikes = data.sum(-1).clip(max=MAX_SPIKES_PER_BIN)
# Each element in this 2D array represents the sum of a group of 64 elements from the reshaped array.
first_spike = (data * np.arange(BINS) + (1 - data) * BINS).min(-1)
last_spike = (data * np.arange(1, BINS + 1)).max(-1)
data = np.stack([n_spikes, first_spike, last_spike], axis=-1)

SPATIAL_DIM = data.shape[0]

train_data, eval_data = data[:, 6 * BINS_PER_MINUTE:], data[:, 3 * BINS_PER_MINUTE:5 * BINS_PER_MINUTE]

# viz = eval_data
# plt.hist(viz.reshape(-1), bins=viz.max() + 1, log=True)
# plt.show()

class ChunksDataset(torch.utils.data.Dataset):
    def __init__(self, data, chunk_len=TEMPORAL_DIM + 1):
        super().__init__()
        self.data = data
        self.chunk_len = chunk_len
    
    def __len__(self):
        # return self.data.shape[1] // self.chunk_len
        return self.data.shape[1] - self.chunk_len + 1
    
    def __getitem__(self, idx):
        # return self.data[:, idx*self.chunk_len:(idx+1)*self.chunk_len]
        return self.data[:, idx : idx + self.chunk_len]


train_data_loader = torch.utils.data.DataLoader(ChunksDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
eval_data_loader = torch.utils.data.DataLoader(ChunksDataset(eval_data), batch_size=BATCH_SIZE)

class TransformerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_ln = torch.nn.LayerNorm(EMBEDDING_DIM)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(EMBEDDING_DIM, 4 * EMBEDDING_DIM),
            torch.nn.GELU(),
            torch.nn.Linear(4 * EMBEDDING_DIM, EMBEDDING_DIM),
        )
        self.attn_spatial = torch.nn.MultiheadAttention(EMBEDDING_DIM, N_HEADS, ATTN_DROPOUT)
        self.attn_temporal = torch.nn.MultiheadAttention(EMBEDDING_DIM, N_HEADS, ATTN_DROPOUT)
    
    def forward(self, embeddings):
        shape = einops.parse_shape(embeddings, "b s t e")
        norm_in = self.in_ln(embeddings)
        ffn_out = self.ffn(norm_in)
        attn_spatial_in = einops.rearrange(norm_in, "b s t e -> s (b t) e")
        attn_temporal_in = einops.rearrange(norm_in, "b s t e -> t (b s) e")
        attn_spatial_out, _ = self.attn_spatial(attn_spatial_in, attn_spatial_in, attn_spatial_in)
        attn_mask = (torch.arange(0, TEMPORAL_DIM)[:, None] < torch.arange(0, TEMPORAL_DIM)[None]).cuda()
        attn_temporal_out, _ = self.attn_temporal(attn_temporal_in, attn_temporal_in, attn_temporal_in, attn_mask=attn_mask)
        attn_spatial_out = einops.rearrange(attn_spatial_out, "s (b t) e -> b s t e", **shape)
        attn_temporal_out = einops.rearrange(attn_temporal_out, "t (b s) e -> b s t e", **shape)
        embeddings = embeddings + ffn_out
        embeddings = embeddings + attn_spatial_out
        embeddings = embeddings + attn_temporal_out
        return embeddings

class TransformerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_emeddings = torch.nn.Parameter(torch.randn((SPATIAL_DIM, EMBEDDING_DIM)))
        self.temporal_emeddings = torch.nn.Parameter(torch.randn((TEMPORAL_DIM, EMBEDDING_DIM)))
        self.input_embeddings_cnt = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, MAX_SPIKES_PER_BIN + 1)))
        self.input_embeddings_first = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, BINS + 1)))
        self.input_embeddings_last = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, BINS + 1)))
        self.output_embeddings_cnt = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, MAX_SPIKES_PER_BIN + 1)))
        self.output_embeddings_first = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, BINS + 1)))
        self.output_embeddings_last = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, BINS + 1)))
        self.transformer = torch.nn.Sequential(*(TransformerBlock() for _ in range(6)))
    
    def forward(self, x):
        embeddings = \
            einops.rearrange(self.spatial_emeddings, "s e -> () s () e") + \
            einops.rearrange(self.temporal_emeddings, "t e -> () () t e") + \
            einops.einsum(F.one_hot(x[:, :, :, 0], MAX_SPIKES_PER_BIN + 1).to(torch.float32), self.input_embeddings_cnt, "b s t S, e S -> b s t e") + \
            einops.einsum(F.one_hot(x[:, :, :, 1], BINS + 1).to(torch.float32), self.input_embeddings_first, "b s t S, e S -> b s t e") + \
            einops.einsum(F.one_hot(x[:, :, :, 2], BINS + 1).to(torch.float32), self.input_embeddings_last, "b s t S, e S -> b s t e")
        embeddings = self.transformer(embeddings)
        return [
            einops.einsum(embeddings, self.output_embeddings_cnt, "b s t e, e S -> b s t S"),
            einops.einsum(embeddings, self.output_embeddings_first, "b s t e, e S -> b s t S"),
            einops.einsum(embeddings, self.output_embeddings_last, "b s t e, e S -> b s t S"),
        ]

model = TransformerModel().cuda()
# model = torch.compile(model)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
# opt_dict, state_dict = torch.load("checkpoints/model_22.pt")
# # state_dict['_orig_mod.temporal_emeddings'] = state_dict['_orig_mod.temporal_emeddings'][:TEMPORAL_DIM]
# model.load_state_dict(state_dict)
# opt.load_state_dict(opt_dict)

np.set_printoptions(precision=5)

def gen(state, total_len):
    state = torch.tensor(state).cuda()
    model.eval()
    pb = tqdm.tqdm(total=total_len - state.shape[1])
    while state.shape[1] < total_len:
        logits_all = model(state[None, :, -TEMPORAL_DIM:])
        samples_all = [torch.distributions.Categorical(logits=logits[0, :, -1:]).sample() for logits in logits_all]
        samples = torch.stack(samples_all, axis=-1)
        state = torch.cat([state, samples], axis=1)
        pb.update(1)
    return state.cpu().numpy()

def correlations(data):
    data = torch.tensor(data).cuda().float()
    std = torch.sqrt(torch.einsum("a t, a t -> a", data, data)).maximum(torch.tensor(1e-8).cuda())
    corr = torch.einsum("a t, b t, a, b -> a b", data, data, 1.0 / std, 1.0 / std)
    return corr.cpu().numpy()

# gen_data = gen(eval_data[:, :TEMPORAL_DIM], total_len=eval_data.shape[1])
# real_corr = correlations(eval_data[:, :, 0])
# gen_corr = correlations(gen_data[:, :, 0])
# plt.hist(real_corr.reshape(-1), bins=100, alpha=0.5, label="real data")
# plt.hist(gen_corr.reshape(-1), bins=100, alpha=0.5, label="LLM generated")
# plt.xlabel("correlation coefficient")
# plt.ylabel("# of pairs")
# plt.legend()
# plt.show()
# plt.plot(eval_data.sum(0), label="real data", alpha=0.5)
# plt.plot(gen_data.sum(0), label="LLM generated", alpha=0.5)
# plt.xlabel("time")
# plt.ylabel("# of spikes")
# plt.legend()
# plt.show()
# os.exit(0)

def train_epoch(epoch, grad_accum=4):
    model.train()
    loss_sum = np.array([0.0] * TEMPORAL_DIM)
    loss_cnt = 0
    for step, batch in enumerate(bar := tqdm.tqdm(iter(train_data_loader), dynamic_ncols=True)):
        batch = batch.cuda()

        logits_all = model(batch[:, :, :-1])
        loss_all = [F.cross_entropy(einops.rearrange(logits, "b s t S -> b S s t"), batch[:, :, 1:, i], reduction="none") for i, logits in enumerate(logits_all)]
        loss = sum(loss_all)
        loss = loss.mean(0).mean(0)

        if step % grad_accum == 0:
            opt.zero_grad()
        loss.mean().backward()
        if step % grad_accum == grad_accum - 1:
            opt.step()

        loss_sum = EMA * loss_sum + (1.0 - EMA) * loss.detach().cpu().numpy()
        loss_cnt = EMA * loss_cnt + (1.0 - EMA)
        bar.set_description(f'Epoch {epoch} train loss: {loss_sum / loss_cnt}')

def eval_epoch(epoch):
    model.eval()
    loss_sum = np.array([0.0] * TEMPORAL_DIM)
    loss_cnt = 0
    with torch.no_grad():
        for batch in (bar := tqdm.tqdm(iter(eval_data_loader), dynamic_ncols=True)):
            batch = batch.cuda()

            logits_all = model(batch[:, :, :-1])
            loss_all = [F.cross_entropy(einops.rearrange(logits, "b s t S -> b S s t"), batch[:, :, 1:, i], reduction="none") for i, logits in enumerate(logits_all)]
            loss = sum(loss_all)
            loss = loss.mean(0).mean(0)

            loss_sum += loss.cpu().numpy()
            loss_cnt += 1
            bar.set_description(f'Epoch {epoch} eval loss: {loss_sum / loss_cnt}')
    plt.figure()
    plt.plot(eval_data[:, :, 0].sum(0), label="real", alpha=0.5)
    gen_data = gen(eval_data[:, :TEMPORAL_DIM], total_len=eval_data.shape[1])
    plt.plot(gen_data[:, :, 0].sum(0), label="gen", alpha=0.5)
    plt.legend()
    # plt.show()
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/eval_{epoch}.png")


for epoch in range(1000):
    train_epoch(epoch)
    if epoch % 1 == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save((opt.state_dict(), model.state_dict()), f"checkpoints/model_{epoch}.pt")
        eval_epoch(epoch)
