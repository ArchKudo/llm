import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):

    def __init__(self, text, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i : i + max_len]
            target_chunk = token_ids[i + 1 : i + max_len + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    drop_last=True,
    shuffle=True,
    num_workers=0,
):

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_length, stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


if __name__ == "__main__":

    with open("./The_Verdict.txt", "r", encoding="utf-8") as f:
        book = "".join(f.read().splitlines()[21:186])

    print(book[:20])
    dataloader = create_dataloader(
        book, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    print(next(data_iter))
