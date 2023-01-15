import os
import random


def rand_scenario_filepath(override_folder:str = None) -> str:
    scenario_folder = "./scenarios" if override_folder is None else override_folder
    sc_dir = os.listdir(scenario_folder)
    selector = random.randint(0, len(sc_dir)-1)
    return f"{scenario_folder}/{sc_dir[selector]}"