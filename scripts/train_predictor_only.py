import tensorflow as tf

import os
import sys
sys.path.insert(0, os.curdir)

from src.mnvr.trainer.trainer import PredictorTrainer, Explorer_One

from src.mnvr.model import get_mk_1, get_mk_2
from src.mnvr.trainer.runner import ManoeverRunner

_, predictor = get_mk_2()


def predictor_run():
    p_t = PredictorTrainer(predictor)
    p_t.train_with_explorer()

    
for n in range(4):
    predictor_run()
