#!/bin/bash
# Monitoring script for multi-node GPU jobs

JOBID=$1

if [ -z "$JOBID" ]; then
    echo "Usage: $0 <job_id>"
    echo "Example: $0 44572948"
    exit 1
fi

echo "=== JOB STATUS ==="
squeue -j $JOBID -o "%.10i %.9P %.20j %.2t %.10M %.6D %N"

echo ""
echo "=== NODELIST ==="
NODELIST=$(squeue -j $JOBID -h -o "%N" 2>/dev/null)
if [ ! -z "$NODELIST" ]; then
    # Use scontrol to expand node list (handles nid[003700,008377] format)
    scontrol show hostnames $NODELIST 2>/dev/null | sort -u
else
    echo "No nodes found"
fi

echo ""
echo "=== GPU USAGE PER NODE ==="
if [ ! -z "$NODELIST" ]; then
    # Expand node list using scontrol
    NODES=$(scontrol show hostnames $NODELIST 2>/dev/null)
    if [ ! -z "$NODES" ]; then
        for node in $NODES; do
            echo "Node: $node"
            ssh $node "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv" 2>/dev/null | head -5 || echo "  Cannot access"
        done
    fi
else
    echo "No nodes found"
fi

echo ""
echo "=== PYTHON PROCESSES ==="
if [ ! -z "$NODELIST" ]; then
    # Expand node list using scontrol
    NODES=$(scontrol show hostnames $NODELIST 2>/dev/null)
    if [ ! -z "$NODES" ]; then
        for node in $NODES; do
            echo "Node: $node"
            ssh $node "ps aux | grep 'main.py' | grep -v grep | wc -l" 2>/dev/null | xargs echo "  Processes:" || echo "  Cannot access"
        done
    fi
else
    echo "No nodes found"
fi

echo ""
echo "=== LATEST ERROR OUTPUT ==="
tail -20 ptycho_debug_${JOBID}.err 2>/dev/null || echo "No error file yet"

echo ""
echo "=== LATEST STDOUT OUTPUT ==="
tail -20 ptycho_debug_${JOBID}.out 2>/dev/null || echo "No output file yet"
