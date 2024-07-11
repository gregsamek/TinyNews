import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm


class TinyNewsDataset(Dataset):
    def __init__(self, text_path, vocab_size, max_length):
        self.tokens = []

        tokenizer = spm.SentencePieceProcessor(model_file=f'tokenizers/tinynewstokenizer{vocab_size}.model')

        with open(text_path, 'r') as f:
            for news in f:
                token_ids = tokenizer.encode(news.replace('\n', ''), out_type=int)
                if len(token_ids) <= (max_length-2): # -1 because we want to end on at least one eos token
                    token_ids = torch.tensor(token_ids, dtype=torch.int)
                    bos = torch.tensor([1], dtype=torch.int) # bos token is token id 1
                    eos = torch.tensor([2], dtype=torch.int) # eos token is token id 2
                    token_ids = torch.cat((bos, token_ids, eos))
                    self.tokens.append(token_ids)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]
    
def collate_fn(batch):
    # Find the maximum length in the batch
    max_length = max(len(sample) for sample in batch)
    
    # Pad sequences to the maximum length
    return torch.stack([torch.cat(
        (sample, torch.full((max_length-len(sample),), 0, dtype=torch.int))) for sample in batch])

def create_dataloader(text_path, vocab_size=8192, batch_size=4, max_length=128, 
                         shuffle=True, num_workers=0):    

    dataset = TinyNewsDataset(text_path, vocab_size, max_length)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)

    return dataloader