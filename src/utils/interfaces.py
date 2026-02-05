from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from typing import Optional, Any, List
from numpy import ndarray
from torch.utils.data import IterableDataset
import math
@dataclass
class LabeledImageData:
    img: torch.Tensor = None
    condition: Any = None
    img_path: str = None
    additional_attr: Optional[dict] = None
    def _to(self, device_or_dtype): # inplace
        if self.img is not None and isinstance(self.img, torch.Tensor):
            self.img = self.img.to(device_or_dtype)
        if (self.condition is not None) and isinstance(self.condition, torch.Tensor):
            self.condition = self.condition.to(device_or_dtype).long() # shame on me! This is so bad
        return self
    def to(self, device_or_dtype):
        data = LabeledImageData(
            img=self.img.to(device_or_dtype) if (self.img is not None) and isinstance(self.img, torch.Tensor) else self.img,
            condition=self.condition.to(device_or_dtype).long() if (self.condition is not None) and isinstance(self.condition, torch.Tensor) else self.condition,
            img_path=self.img_path,
            additional_attr=self.additional_attr
        )
        return data
    def __getitem__(self, idx):
        return LabeledImageData(
            img=self.img[idx] if (self.img is not None) and isinstance(self.img, torch.Tensor) else self.img,
            condition=self.condition[idx] if (self.condition is not None) and isinstance(self.condition, torch.Tensor) else self.condition,
            img_path=self.img_path[idx] if self.img_path is not None else self.img_path,
            additional_attr=self.additional_attr[idx] if self.additional_attr is not None else self.additional_attr
        )
    def __len__(self):
        img_len = len(self.img) if (self.img is not None) and isinstance(self.img, torch.Tensor) else 1
        condition_len = len(self.condition) if (self.condition is not None) and isinstance(self.condition, torch.Tensor) else 1
        return max(img_len, condition_len)
    def __setitem__(self, idx, value):
        raise ValueError('Cannot set item to LabeledImageData')

"""
class IterableDatasetShard(IterableDataset):
    Wraps a PyTorch `IterableDataset` to generate samples for one of the processes only. Instances of this class will
    always yield a number of samples that is a round multiple of the actual batch size (which is `batch_size x
    num_processes`). Depending on the value of the `drop_last` attribute, it will either stop the iteration at the
    first batch that would be too small or loop with indices from the beginning.

    On two processes with an iterable dataset yielding of `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` with a batch size of
    2:

    - the shard on process 0 will yield `[0, 1, 4, 5, 8, 9]` so will see batches `[0, 1]`, `[4, 5]`, `[8, 9]`
    - the shard on process 1 will yield `[2, 3, 6, 7, 10, 11]` so will see batches `[2, 3]`, `[6, 7]`, `[10, 11]`

    <Tip warning={true}>

        If your IterableDataset implements some randomization that needs to be applied the same way on all processes
        (for instance, a shuffling), you should use a `torch.Generator` in a `generator` attribute of the `dataset` to
        generate your random numbers and call the [`~trainer_pt_utils.IterableDatasetShard.set_epoch`] method of this
        object. It will set the seed of this `generator` to `seed + epoch` on all processes before starting the
        iteration. Alternatively, you can also implement a `set_epoch()` method in your iterable dataset to deal with
        this.

    </Tip>

    Args:
        dataset (`torch.utils.data.IterableDataset`):
            The batch sampler to split in several shards.
        batch_size (`int`, *optional*, defaults to 1):
            The size of the batches per shard.
        drop_last (`bool`, *optional*, defaults to `False`):
            Whether or not to drop the last incomplete batch or complete the last batches by using the samples from the
            beginning.
        num_processes (`int`, *optional*, defaults to 1):
            The number of processes running concurrently.
        process_index (`int`, *optional*, defaults to 0):
            The index of the current process.
        seed (`int`, *optional*, defaults to 0):
            A random seed that will be used for the random number generation in
            [`~trainer_pt_utils.IterableDatasetShard.set_epoch`].

    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int = 1,
        drop_last: bool = False,
        num_processes: int = 1,
        process_index: int = 0,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_processes = num_processes
        self.process_index = process_index
        self.seed = seed
        self.epoch = 0
        self.num_examples = 0
        if hasattr(self.dataset, "collate_fn"):
            self.collate_fn = self.dataset.collate_fn # override the collate_fn
    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __iter__(self):
        self.num_examples = 0
        if (
            not hasattr(self.dataset, "set_epoch")
            and hasattr(self.dataset, "generator")
            and isinstance(self.dataset.generator, torch.Generator)
        ):
            self.dataset.generator.manual_seed(self.seed + self.epoch)
        real_batch_size = self.batch_size * self.num_processes
        process_slice = range(self.process_index * self.batch_size, (self.process_index + 1) * self.batch_size)

        first_batch = None
        current_batch = []
        for element in self.dataset:
            self.num_examples += 1
            current_batch.append(element)
            # Wait to have a full batch before yielding elements.
            if len(current_batch) == real_batch_size:
                for i in process_slice:
                    yield current_batch[i]
                if first_batch is None:
                    first_batch = current_batch.copy()
                current_batch = []

        # Finished if drop_last is True, otherwise complete the last batch with elements from the beginning.
        if not self.drop_last and len(current_batch) > 0:
            if first_batch is None:
                first_batch = current_batch.copy()
            while len(current_batch) < real_batch_size:
                current_batch += first_batch
            for i in process_slice:
                yield current_batch[i]
    
    def __len__(self):
        # Will raise an error if the underlying dataset is not sized.
        if self.drop_last:
            return (len(self.dataset) // (self.batch_size * self.num_processes)) * self.batch_size
        else:
            return math.ceil(len(self.dataset) / (self.batch_size * self.num_processes)) * self.batch_size
"""

def LabeledImageDatasetWrapper(dataset: Dataset) -> Dataset:
    if isinstance(dataset, IterableDataset):
        base = IterableDataset
    else:
        base = Dataset
    class LabeledImageDatasetWrapper(base):
        """
        We assume the base dataset returns a tuple of (img, label, img_path, additional_attr), the last two being optional.
        """
        def __init__(self, dataset: Dataset):
            self.dataset = dataset
            if hasattr(dataset, 'collate_fn'):
                self.collate_fn = dataset.collate_fn # override the collate_fn
            else:
                self.collate_fn = self.default_collate_fn
            # see if dataset has __getitem__ method
            if base == Dataset:
                pass
                #self.__getitem__ = self.potential_getitem # register __getitem__ method
                #setattr(self, '__getitem__', self.potential_getitem)
        def __len__(self):
            return len(self.dataset)
        def set_epoch(self, epoch: int):
            # if dataset has set_epoch method, call it
            if hasattr(self.dataset, 'set_epoch'):
                self.dataset.set_epoch(epoch)
            # else donothing
        def __iter__(self):
            for item in self.dataset:
                if isinstance(item, LabeledImageData):
                    yield item
                elif isinstance(item, tuple):
                    yield LabeledImageData(*item)
                elif isinstance(item, dict):
                    yield LabeledImageData(**item)
                elif isinstance(item, torch.Tensor):
                    yield LabeledImageData(img=item)
                else:
                    yield ValueError(f'Unsupported return type from base dataset: {type(item)}')
        #def __getitem__(self, idx):
        #    raise ValueError('This dataset is not indexable')
        def __getitem__(self, idx):
            item = self.dataset[idx]
            if isinstance(item, LabeledImageData):
                return item
            elif isinstance(item, tuple):
                return LabeledImageData(*item)
            elif isinstance(item, dict):
                return LabeledImageData(**item)
            elif isinstance(item, torch.Tensor):
                return LabeledImageData(img=item)
            else:
                raise ValueError(f'Unsupported return type from base dataset: {type(item)}')
        def default_collate_fn(self, batch: List[LabeledImageData]) -> LabeledImageData:
            img, condition, img_path, additional_attr = zip(*[(x.img, x.condition, x.img_path, x.additional_attr) for x in batch])
            # if condition is tensor then stack
            if condition[0] is not None:
                if isinstance(condition[0], torch.Tensor):
                    condition = torch.stack(condition)
                elif isinstance(condition[0], int) or isinstance(condition[0], float) or isinstance(condition[0], bool) or isinstance(condition[0], ndarray):
                    condition = torch.Tensor(condition)
            img = torch.stack(img)
            return LabeledImageData(img=img, condition=condition, img_path=img_path, additional_attr=additional_attr)
    return LabeledImageDatasetWrapper(dataset)
