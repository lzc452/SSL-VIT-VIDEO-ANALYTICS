from torch.utils.data import Dataset


class BaseVideoDataset(Dataset):
    """
    所有视频 Dataset 的统一抽象接口
    """

    def __init__(self, split_file, mode="ssl"):
        self.split_file = split_file
        self.mode = mode

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
