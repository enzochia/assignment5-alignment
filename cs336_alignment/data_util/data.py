import os
import json
from torch.utils.data import Dataset


class MATH_SFT_Dataset(Dataset):
    def __init__(
        self,
        data_path: str | os.PathLike = "data/MATH/sft.jsonl"
    ):
        super().__init__()
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        # self.data = self.data[:20]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    outputs = [item["response"] for item in batch]
    answers = [item["ground_truth"] for item in batch]
    return prompts, outputs, answers