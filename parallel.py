from typing import Any, Optional
import os
import re
import socket
import subprocess

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
        
        # Map SLURM/PMI environment variables to PyTorch distributed variables
        # This ensures compatibility with both SLURM srun and Cray MPI systems
        env = os.environ
        
        # Handle SLURM variables (set by srun)
        if "SLURM_PROCID" in env and "RANK" not in env:
            os.environ["RANK"] = env["SLURM_PROCID"]
        
        if "SLURM_NTASKS" in env and "WORLD_SIZE" not in env:
            os.environ["WORLD_SIZE"] = env["SLURM_NTASKS"]
        
        if "SLURM_LOCALID" in env and "LOCAL_RANK" not in env:
            os.environ["LOCAL_RANK"] = env["SLURM_LOCALID"]
        
        # Handle PMI variables (Cray MPI/PMI interface)
        if "PMI_LOCAL_RANK" in env and "LOCAL_RANK" not in env:
            os.environ["LOCAL_RANK"] = env["PMI_LOCAL_RANK"]
        
        if "PMI_RANK" in env and "RANK" not in env:
            os.environ["RANK"] = env["PMI_RANK"]
        
        if "PMI_SIZE" in env and "WORLD_SIZE" not in env:
            os.environ["WORLD_SIZE"] = env["PMI_SIZE"]
        
        # Set MASTER_ADDR and MASTER_PORT if not already set
        # For multi-node jobs, use the first node in SLURM_JOB_NODELIST
        if "MASTER_ADDR" not in env:
            if "SLURM_JOB_NODELIST" in env:
                # Use scontrol to expand node list (handles nid[003700,008377] format)
                nodelist = env["SLURM_JOB_NODELIST"]
                try:
                    result = subprocess.run(
                        ["scontrol", "show", "hostnames", nodelist],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        # Get first node from expanded list
                        first_node = result.stdout.strip().split('\n')[0]
                        os.environ["MASTER_ADDR"] = first_node
                    else:
                        # Fallback: use current hostname
                        os.environ["MASTER_ADDR"] = socket.gethostname()
                except Exception:
                    # Fallback: use current hostname if scontrol fails
                    os.environ["MASTER_ADDR"] = socket.gethostname()
            else:
                os.environ["MASTER_ADDR"] = socket.gethostname()
        
        if "MASTER_PORT" not in env:
            os.environ["MASTER_PORT"] = "29500"
        
        dist.init_process_group(backend=backend, init_method="env://")

    def detect_launcher(self) -> str | None:
        env = os.environ
        if "GROUP_RANK" in env or "ROLE_RANK" in env or "LOCAL_WORLD_SIZE" in env:
            return "torchrun"
        elif "RANK" in env and "WORLD_SIZE" in env and "LOCAL_RANK" in env:
            return "launch"
        elif "SLURM_PROCID" in env or "PMI_LOCAL_RANK" in env:
            # SLURM or PMI (Cray MPI) detected - will set env vars in init_process_group
            return "launch"
        else:
            return None
