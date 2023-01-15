
def _yparse(filepath:str) -> dict:
        import yaml
        from beamngpy import Scenario
        with open(filepath, 'r') as file:
            obj = yaml.load(file, yaml.FullLoader)
            file.close()
        return obj

class MnvrScenario:
    import beamngpy
    def __init__(self, filepath:str):
        data = _yparse(filepath)
        
        self.level = data['level']
        self.name = data['name']
        self.path = data['path']
        self.start_pos = data['start_pos']
        self.start_rot = data['start_rot']
        
    def get(self) -> tuple[beamngpy.Scenario, tuple]:
        from beamngpy import Scenario
        try:
            scenario = Scenario(
                level=self.level,
                name=self.name,
                path=self.path
            )
            
            return scenario, (self.start_pos, self.start_rot)
        except Exception as e:
            raise(e)

class MnvrConfig:
    def __init__(self, filepath:str = "./mnvr.yaml") -> None:
        import yaml
        data = _yparse(filepath)
        
        self.user_folder = data['user_folder']
        self.runtime_host = data['runtime_host']
        self.runtime_port = data['runtime_port']
        self.beam_installation = data['beam_installation']
        self.runtime_host = data['runtime_host']
        self.car_path = data['car_path']
        self.sim_hz = data['simulation_hz']
            
    def get_prebaked_beamng(self):
        import beamngpy
        bng = beamngpy.BeamNGpy(
            host=self.runtime_host,
            port=self.runtime_port,
            home=self.beam_installation,
            user=self.user_folder
        )
        
        return bng
