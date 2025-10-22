from typing import Any, Optional
import os

import torch
import torch.distributed as dist
from torch import Tensor
import numpy as np


class MultiprocessMixin:
    backend = "nccl"
    
    @property
    def rank(self) -> int:
        try:
            return dist.get_rank()
        except ValueError:
            return 0
    
    @property
    def n_ranks(self) -> int:
        try:
            return dist.get_world_size()
        except ValueError:
            return 1
    
    def get_chunk_of_current_rank(self, tensor: Tensor) -> Tensor:
        """
        Get a chunk of the tensor for the current rank.
        
        Parameters
        ----------
        tensor : Tensor
            The tensor to be chunked. The first dimension of the tensor is assumed
            to be the batch dimension and it will be split along this dimension.
            
        Returns
        -------
        Tensor
            A chunk of the tensor for the current rank.
        """
        chunks = torch.chunk(tensor, self.n_ranks, dim=0)
        if len(chunks) != self.n_ranks:
            # torch.chunk might return fewer chunks than asked in some cases.
            chunk_size = tensor.shape[0] // self.n_ranks
            start = self.rank * chunk_size
            end = min(start + chunk_size, tensor.shape[0])
            return tensor[start:end]
        else:
            return chunks[self.rank]

    def init_process_group(self, backend: str = "nccl") -> None:
        if dist.is_initialized():
            return
        dist.init_process_group(backend=backend, init_method="env://")

    def detect_launcher(self) -> str | None:
        env = os.environ
        if "GROUP_RANK" in env or "ROLE_RANK" in env or "LOCAL_WORLD_SIZE" in env:
            return "torchrun"
        elif "RANK" in env and "WORLD_SIZE" in env and "LOCAL_RANK" in env:
            return "launch"
        else:
            return None
