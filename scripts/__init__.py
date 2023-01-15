import os
import sys
sys.path.insert(0, os.curdir)

from src.mnvr.utils import setup_required

def run_startup_tasks():
    setup_required()