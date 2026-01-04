import os
import json
from torch.utils.data import Dataset
# from cs336_alignment.grpo import GRPOConfig
# from cs336_alignment.sft.configs import SFTConfig


class Train_Dataset(Dataset):
    def __init__(
        self,
        configs,
        train: str = "grpo"
    ):
    # TODO: if import the config classes there would be circular imports. fix it.
        super().__init__()
        self.configs = configs
        self.data = []
        paths = None
        if train == "grpo":
            paths = configs.data_train_path
        elif train == "sft":
            paths = configs.data_sft_path
        for path in paths:
            with open(path, "r") as f:
                for line in f:
                    self.data.append(json.loads(line))
        # self.data = self.data[:20]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn_sft(self, batch):
        if self.configs.sft_data_name == "MATH":
            prompts = [item["prompt"] for item in batch]
            outputs = [item["response"] for item in batch]
            answers = [item["ground_truth"] for item in batch]
            return prompts, outputs, answers
        else:
            raise ValueError(f"Unsupported SFT data: {self.configs.train_data_name}")

    def collate_fn_train(self, batch):
        if self.configs.train_data_name == "MATH":
            problem = [item["problem"] for item in batch]
            level = [item["level"] for item in batch]
            subject = [item["subject"] for item in batch]
            unique_id = [item["unique_id"] for item in batch]
            answer = [item["answer"] for item in batch]
            return problem, level, subject, unique_id, answer
        elif self.configs.train_data_name == "gsm8k":
            problem = [item["question"] for item in batch]
            answer = [item["answer"] for item in batch]
            return problem, [], [], [], answer
