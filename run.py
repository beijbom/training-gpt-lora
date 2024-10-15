import argparse
import os
import time
from io import BytesIO
from typing import Callable

import modal
import torch

from data import get_batch_factory, get_encoder_decoder_vocab_size
from gpt import GPTLanguageModel, batch_size, block_size

device = "cuda" if torch.cuda.is_available() else "cpu"

# Modal configuration
lora_image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")
app = modal.App("lora-tutorial")


@torch.no_grad()
def estimate_loss(model: GPTLanguageModel, get_batch: Callable, eval_iters: int):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@app.function(image=lora_image, gpu="A100", timeout=3600)
def train(
    files: dict[str, BytesIO],
    max_iters: int = 5000,
    eval_interval: int = 500,
    learning_rate: float = 3e-4,
) -> dict[str, BytesIO]:

    encode, _, vocab_size = get_encoder_decoder_vocab_size(files["shakespeare.txt"])

    model = GPTLanguageModel(vocab_size)
    model = model.to(device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, "M trainable parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    get_batch = get_batch_factory(files["shakespeare.txt"], block_size, batch_size, device, encode)
    start_time = time.time()
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, get_batch, eval_iters)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, timer {time.time() - start_time:.2f}s"
            )

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the model weights
    buff = BytesIO()
    torch.save(model.state_dict(), buff)
    return {"shakespeare.pth": buff}


@app.function(image=lora_image, gpu="A100")
def generate(
    files: dict[str, BytesIO],
    base_model_checkpoint: str,
    lora_model_checkpoint: str | None = None,
) -> dict[str, BytesIO]:
    _, decode, vocab_size = get_encoder_decoder_vocab_size(files["shakespeare.txt"])

    model = GPTLanguageModel(vocab_size)
    model.load_state_dict(torch.load(files[base_model_checkpoint]), strict=False)
    if lora_model_checkpoint:
        model.load_state_dict(torch.load(files[lora_model_checkpoint]), strict=False)
    model.eval()
    model.to(device)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

    more = decode(model.generate(context, max_new_tokens=5000)[0].tolist())
    buff = BytesIO()
    buff.write(more.encode("utf-8"))
    return {"more.txt": buff}


@app.function(image=lora_image, gpu="A100")
def tune(
    files: dict[str, BytesIO],
    max_iters: int = 300,
    eval_interval: int = 100,
    eval_iters: int = 200,
    learning_rate: float = 1e-4,
):
    encode, _, vocab_size = get_encoder_decoder_vocab_size(files["shakespeare.txt"])

    model = GPTLanguageModel(vocab_size)
    model.load_state_dict(torch.load(files["shakespeare.pth"]), strict=False)
    model = model.to(device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    # create a PyTorch optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    get_batch = get_batch_factory(files["hemingway.txt"], block_size, batch_size, device, encode)

    start_time = time.time()
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, get_batch, eval_iters)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, timer {time.time() - start_time:.2f}s"
            )

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the model weights
    buff = BytesIO()
    torch.save(model.state_dict(), buff)
    return {"hemingway.pth": buff}


class FileSyncer:
    files_to_sync = [
        "shakespeare.txt",
        "hemingway.txt",
        "shakespeare.pth",
        "hemingway.pth",
        "hemingway_lora.pth",
        "more-shakespeare.txt",
    ]

    @classmethod
    def load(cls) -> dict[str, BytesIO]:
        files: dict[str, BytesIO] = {}
        for filename in cls.files_to_sync:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    files[filename] = BytesIO(f.read())
        return files

    @classmethod
    def store(cls, files: dict[str, BytesIO]) -> None:
        for filename, data in files.items():
            with open(filename, "wb") as f:
                f.write(data.getbuffer())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--base_model_checkpoint", type=str, default="shakespeare.pth")
    parser.add_argument("--lora_model_checkpoint", type=str, default=None)
    args = parser.parse_args()

    input_files = FileSyncer.load()

    with modal.enable_output():
        with app.run():
            if args.mode == "train":
                output_files = train.remote(input_files)
            elif args.mode == "generate":
                output_files = generate.remote(input_files, args.base_model_checkpoint, args.lora_model_checkpoint)
                print("Done generating.")
            elif args.mode == "tune":
                output_files = tune.remote(input_files)
    FileSyncer.store(output_files)
