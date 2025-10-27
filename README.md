Parallel ptychography simulator based on Pty-Chi.

To run batch simulation, use
```
python main /path/to/config.yaml
```
Refer to the sample config YAML file for available settings. 

To launch multiple processes, use
```
torchrun --nnodes 1 --nproc-per-node 2 main /path/to/config.yaml
```
