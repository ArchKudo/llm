# import tiktoken
# import torch
# from torch.utils.data import Dataset, DataLoader


# class GPTDataset(Dataset):
#     def __init__(self, text, tokenizer, max_len, stride):
#         self.input_ids = []
#         self.target_ids = []

#         token_ids = tokenizer.encode(text)

#         for i in range(0, len(token_ids) - max_len, stride):
#             input_chunk = token_ids[i : i + max_len]
#             target_chunk = token_ids[i + 1 : i + max_len + 1]

#             self.input_ids.append(torch.tensor(input_chunk))
#             self.target_ids.append(torch.tensor(target_chunk))

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, idx):
#         return self.input_ids[idx], self.target_ids[idx]


# def create_dataloader(
#     text,
#     batch_size=4,
#     max_length=256,
#     stride=128,
#     drop_last=True,
#     shuffle=True,
#     num_workers=0,
# ):

#     tokenizer = tiktoken.get_encoding("gpt2")
#     dataset = GPTDataset(text, tokenizer, max_length, stride)

#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         drop_last=drop_last,
#         num_workers=num_workers,
#     )


# if __name__ == "__main__":
#     with open("./The_Verdict.txt", "r", encoding="utf-8") as f:
#         book = "".join(f.read().splitlines()[21:186])

#     print(book[:20])
#     dataloader = create_dataloader(
#         book, batch_size=1, max_length=4, stride=1, shuffle=False
#     )

#     data_iter = iter(dataloader)
#     first_batch = next(data_iter)
#     print(first_batch)
#     print(next(data_iter))

#     dataloader = create_dataloader(
#         book, batch_size=8, max_length=4, stride=4, shuffle=False
#     )

#     data_iter = iter(dataloader)
#     print(next(data_iter))
#     print(next(data_iter))

#     vocab_size = 50257
#     output_dim = 256

#     token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

#     dataloader = create_dataloader(
#         book, batch_size=8, max_length=4, stride=4, shuffle=False
#     )

#     data_iter = iter(dataloader)
#     inputs, targets = next(data_iter)

#     token_embeddings = token_embedding_layer(inputs)
#     print(token_embeddings)
#     print(token_embeddings.shape)

#     context_length = 4
#     pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
#     pos_embeddings = pos_embedding_layer(torch.arange(context_length))

#     print(pos_embeddings)
#     print(pos_embeddings.shape)

#     # Broadcast pos_embeddings [1,4,256] -> [8,4,256]
#     input_embeddings = token_embeddings + pos_embeddings

#     print(input_embeddings)
#     print(input_embeddings.shape)
