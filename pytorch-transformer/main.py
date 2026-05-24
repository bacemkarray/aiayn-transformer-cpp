import torch

from model import Transformer


def main() -> None:
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    batch_size = 2
    src_seq_len = 12
    tgt_seq_len = 10

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=128,
        num_heads=8,
        num_layers=2,
        d_ff=512,
    )

    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    logits = model(src, tgt)

    print(f"Transformer output shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
