import os
import sys
sys.path.insert(0, os.curdir)

from src.mnvr.trainer import (
    ManoeverRunner,
    rand_scenario_filepath
)

from src.mnvr.model import get_mk_2


def simulate(agent):
    runner = ManoeverRunner()
    scenario_path = rand_scenario_filepath()
    runner.preload_scenario(scenario_path)
    results = runner.run_simulator(agent)
    
    del runner
    return results

if __name__ == '__main__':
    agent, _ = get_mk_2()
    simulate(agent)