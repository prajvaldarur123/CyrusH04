#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
export PYTHONUNBUFFERED=1
cd /mnt/c/Users/prajval/robotics/cyrus
/opt/venv_mjx/bin/python training/train_balance.py
