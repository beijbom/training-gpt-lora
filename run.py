from data import get_batch_factory
from gpt import GPTLanguageModel
import torch
from typing import Callable
import fire
from data import get_encoder_decoder_vocab_size

# hyperparameters
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
# ------------


@torch.no_grad()
def estimate_loss(model: GPTLanguageModel, get_batch: Callable):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():

    model = GPTLanguageModel()
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    get_batch = get_batch_factory('shakespeare.txt', block_size, batch_size, device)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, get_batch)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the model weights
    torch.save(model.state_dict(), 'shakespeare_weights.pth')
    print("Model weights saved successfully.")

def generate(model_checkpoint: str = 'shakespeare_weights.pth'):
    model = GPTLanguageModel()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.to(device)

    _, decode, _ = get_encoder_decoder_vocab_size()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

if __name__ == '__main__':

    fire.Fire({
        'train': train,
        'generate': generate
    })