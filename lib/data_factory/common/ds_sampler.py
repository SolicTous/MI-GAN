import torch
import torch.distributed as dist

from ...log_service import print_log


class DistributedSampler(torch.utils.data.Sampler):
    """
    This is a new DistributedSampler that use more sophisticate way
        for random sync.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        extend=False
    ):
        """
        Args:
            extend: bool,
                True, even the number of samples between processes
                    by extending it.
                False, even by truncate it.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise ValueError
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise ValueError
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        # Обработка случая когда датасет пустой или очень маленький
        dataset_len = len(dataset)
        if dataset_len == 0:
            num_samples = 0
            total_size = 0
        else:
            num_samples = dataset_len // num_replicas
            if extend:
                if dataset_len != num_samples*num_replicas:
                    num_samples += 1
            total_size = num_samples * num_replicas

        self.num_samples = num_samples
        self.total_size = total_size
        self.shuffle = shuffle
        self.extend = extend

    def __iter__(self):
        if len(self.dataset) == 0:
            return iter([])
        indices = self.get_sync_order()
        if self.extend:
            # extend using the front indices
            indices = indices+indices[0:self.total_size-len(indices)]
        else:
            # truncate
            indices = indices[0:self.total_size]
        # subsample
        indices = indices[self.rank:len(indices):self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        # legacy
        pass

    def get_sync_order(self):
        if len(self.dataset) == 0:
            return []
        if self.shuffle:
            indices = torch.randperm(len(self.dataset))
            # Для single GPU (world_size=1) broadcast не нужен, но безопасен
            if dist.get_world_size() > 1:
                indices = indices.to(self.rank)
                dist.broadcast(indices, src=0)
                indices = indices.to('cpu')
            indices = indices.tolist()
        else:
            indices = list(range(len(self.dataset)))
        # print_log(str(indices[0:5]))
        return indices
