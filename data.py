from io import BytesIO
from typing import Callable

import torch


def get_encoder_decoder_vocab_size(dataset_file: BytesIO):
    text = dataset_file.getvalue().decode("utf-8")

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"{vocab_size=}")
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: "".join([itos[i] for i in l])  # decoder: take a list of integers, output a string
    return encode, decode, vocab_size


def get_batch_factory(
    dataset_file: BytesIO, block_size: int, batch_size: int, device: str, encode: Callable
) -> Callable:
    text = dataset_file.getvalue().decode("utf-8")
    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    # data loading
    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    return get_batch
