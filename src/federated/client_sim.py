from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from datasets.loader import VideoClipDataset
from federated.utils_fed import encode_clip

# 每个客户端有独立的 DataLoader，local_train 会从全局权重开始做若干 local epoch，然后返回更新后的 state_dict 以及样本数。

class ClientTrainer:
    def __init__(
        self,
        client_id,
        split_path,
        clip_len,
        image_size,
        batch_size,
        num_workers,
        optimizer_cfg,
        local_epochs=1,
        amp_enable=True,
    ):
        self.client_id = client_id
        self.split_path = Path(split_path)
        self.clip_len = clip_len
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.optimizer_cfg = optimizer_cfg
        self.local_epochs = local_epochs
        self.amp_enable = amp_enable

        self._build_loader()

    def _build_loader(self):
        dataset = VideoClipDataset(
            str(self.split_path),
            mode="supervised",
            clip_len=self.clip_len,
            image_size=self.image_size
        )
        self.num_samples = len(dataset)
        self.loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _build_optimizer(self, backbone, head):
        params = list(backbone.parameters()) + list(head.parameters())
        name = self.optimizer_cfg["name"].lower()
        if name == "adamw":
            opt = torch.optim.AdamW(
                params,
                lr=self.optimizer_cfg["lr"],
                weight_decay=self.optimizer_cfg["weight_decay"],
            )
        else:
            raise NotImplementedError(f"Optimizer {name} not implemented in ClientTrainer.")
        return opt

    def local_train(self, backbone, head, device):
        """
        使用本地数据在客户端执行若干 local epochs.
        直接在传入的 backbone/head 上进行更新，传出 state_dict 和 num_samples.
        """
        backbone.train()
        head.train()

        optimizer = self._build_optimizer(backbone, head)
        scaler = GradScaler(enabled=self.amp_enable)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.local_epochs):
            running_loss = 0.0
            for clips, labels in self.loader:
                clips = clips.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with autocast(enabled=self.amp_enable):
                    clip_feat = encode_clip(backbone, clips)    # [B, D]
                    logits = head(clip_feat.unsqueeze(-1).unsqueeze(-1))
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

        return backbone.state_dict(), head.state_dict(), self.num_samples
