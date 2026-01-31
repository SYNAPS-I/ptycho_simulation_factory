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
        launcher = self.detect_launcher()
        
        if launcher == "slurm":
            # Handle SLURM variables (set by srun)
            if "SLURM_PROCID" in env and "RANK" not in env:
                os.environ["RANK"] = env["SLURM_PROCID"]
            
            if "SLURM_NTASKS" in env and "WORLD_SIZE" not in env:
                os.environ["WORLD_SIZE"] = env["SLURM_NTASKS"]
            
            if "SLURM_LOCALID" in env and "LOCAL_RANK" not in env:
                os.environ["LOCAL_RANK"] = env["SLURM_LOCALID"]
        
        elif launcher == "pmi":
            # Handle PMI variables (Cray MPI/PMI interface)
            if "PMI_LOCAL_RANK" in env and "LOCAL_RANK" not in env:
                os.environ["LOCAL_RANK"] = env["PMI_LOCAL_RANK"]
            
            if "PMI_RANK" in env and "RANK" not in env:
                os.environ["RANK"] = env["PMI_RANK"]
            
            if "PMI_SIZE" in env and "WORLD_SIZE" not in env:
                os.environ["WORLD_SIZE"] = env["PMI_SIZE"]
            
            if (
                "PMI_LOCAL_RANK" not in env
                or "PMI_RANK" not in env
                or "PMI_SIZE" not in env
            ):
                try:
                    from mpi4py import MPI
                except ImportError as exc:
                    raise RuntimeError(
                        "mpi4py is required to determine rank/world size when PMI env "
                        "variables are missing."
                    ) from exc
                
                comm = MPI.COMM_WORLD
                os.environ["RANK"] = str(comm.Get_rank())
                os.environ["WORLD_SIZE"] = str(comm.Get_size())
                
                gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 0
                local_rank = comm.Get_rank() % gpus_per_node if gpus_per_node > 0 else 0
                os.environ["LOCAL_RANK"] = str(local_rank)
        
        # Set MASTER_ADDR and MASTER_PORT if not already set
        if "MASTER_ADDR" not in env:
            os.environ["MASTER_ADDR"] = self.get_master_addr(launcher)
        
        if "MASTER_PORT" not in env:
            os.environ["MASTER_PORT"] = "29500"
        
        dist.init_process_group(backend=backend, init_method="env://")

    def get_master_addr(self, launcher: str | None) -> str:
        # For multi-node SLURM jobs, use the first node in SLURM_JOB_NODELIST
        env = os.environ
        if launcher == "slurm" and "SLURM_JOB_NODELIST" in env:
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
                    return result.stdout.strip().split('\n')[0]
            except Exception:
                pass
        # Fallback: use current hostname
        return socket.gethostname()

    def detect_launcher(self) -> str | None:
        env = os.environ
        if "GROUP_RANK" in env or "ROLE_RANK" in env or "LOCAL_WORLD_SIZE" in env:
            return "torchrun"
        elif "RANK" in env and "WORLD_SIZE" in env and "LOCAL_RANK" in env:
            return "launch"
        elif "SLURM_PROCID" in env:
            # SLURM detected - will set env vars in init_process_group
            return "slurm"
        elif "PMI_LOCAL_RANK" in env or "PMI_RANK" in env or "PMI_SIZE" in env:
            # PMI (Cray MPI) detected - will set env vars in init_process_group
            return "pmi"
        else:
            return None
