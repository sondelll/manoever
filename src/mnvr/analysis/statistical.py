import numpy as np
import json
import pandas as pd
import os

from os import listdir

class TrainingProgress:
    def __init__(
        self,
        penalties:list[float],
        distances:list[float],
        damages:list[float]
    ) -> None:
        self.penalties = penalties
        self.distances = distances
        self.damages = damages

    def save_csv(self, path:str):
        ps = pd.Series(self.penalties)
        dists = pd.Series(self.distances)
        dmgs = pd.Series(self.damages)
        data = pd.DataFrame.from_dict({"penalty":ps, "distance":dists, "damage":dmgs})
        data.to_csv(path)


def aggregate_runs_from_dir(runreports_dir:str = "./data/reports/runs"):
    filepaths = listdir(runreports_dir)
    d = runreports_dir
    pen = []
    dist = []
    dmg = []
    for path in filepaths:
        fullpath = f"{d}/{path}"
        if not os.path.isfile(fullpath):
            continue
        with open(fullpath) as file:
            try:
                loaded = json.load(file)
            except:
                continue
            pen.append(loaded['penalty'])
            dist.append(loaded['distance'])
            dmg.append(loaded['damage'])
    return TrainingProgress(pen, dist, dmg)

def process_data(data:dict):
    import numpy as np
    cols=[]
    d=[]
    for k, v in data.items():
        cols.append(k)
        d.append(v)
    d = np.transpose(d)
    return (pd.DataFrame(d, columns=cols), cols)
    
