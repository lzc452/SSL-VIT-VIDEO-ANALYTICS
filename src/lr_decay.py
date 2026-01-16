import math

class CosineScheduler:
    """
    带热身的余弦学习率调度器
    """
    def __init__(self, optimizer, base_lr, total_epochs, warmup_epochs=0, min_lr=0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # 线性热身阶段
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # 余弦衰减阶段
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr