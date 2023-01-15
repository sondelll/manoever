import tensorflow as tf

# A bit hacky but, yes..
import os
import sys
sys.path.insert(0, os.curdir)

from src.mnvr.trainer import (
    AgentTrainer,
    PredictorTrainer,
    ManoeverRunner,
    rand_scenario_filepath
)

from src.mnvr.model import get_mk_1, get_mk_2

from src.mnvr.io.reporter import JsonRunReport

#region Models

agent, predictor = get_mk_2() #get_mk_1()

if agent.built:
    agent.summary()
if predictor.built:
    predictor.summary()

#endregion Models

#region function definitions

def simulate(agent):
    runner = ManoeverRunner()
    scenario_path = rand_scenario_filepath()
    runner.preload_scenario(scenario_path)
    results = runner.run_simulator(agent)
    
    del runner
    return results

def train(agent, predictor):
    a_t = AgentTrainer(agent)
    
    p_t = PredictorTrainer(predictor)
    
    results = simulate(agent)
    
    report = JsonRunReport(results, agent, predictor)
    report.save()
    
    p_t.train_from_results(results)
    
    a_t.train_from_results(results)

    
#endregion

#region Trainability selection

agent.conv_1.trainable = True
agent.conv_2.trainable = True
agent.conv_3.trainable = True
agent.steering.trainable = True
agent.longitude.trainable = True
agent.eb.trainable = True

#endregion


for n in range(8):
    train(agent, predictor)
