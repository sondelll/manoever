import tensorflow as tf

from src.mnvr.trainer.runner import ManoeverRunner

from src.mnvr.trainer.trainer import (
    AgentTrainer,
    Explorer_One,
    PredictorTrainer
)

from src.mnvr.model.m_series import M31
from src.mnvr.model.p_series import P6



#region Models

exploration_agent = Explorer_One()
    
try:
    predictor = tf.keras.models.load_model("./data/saved_models/P6") #PP6()
except:
    predictor = P6()

#endregion Models

#region fn defs

def simulate():
    runner = ManoeverRunner()
    runner.preload_scenario("./scenarios/1A.sc.yaml")
    results = runner.run_simulator()
    
    return results

def train(predictor):
    p_t = PredictorTrainer(predictor)
    
    results = simulate()
    
    p_t.train_from_results(results)
    
#endregion



train(predictor)
