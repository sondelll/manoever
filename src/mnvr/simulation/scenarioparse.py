
class ScenarioParseV1:
    def __init__(self):
        pass
    
    def from_sc_file(self, filepath:str):
        from beamngpy import Scenario
        data = self._yparse(filepath)
        try:
            scenario = Scenario(
                level=data['level'],
                name=data['name'],
                path=data['path']
            )
            return scenario, (data['start_pos'], data['start_rot'])
        except Exception as e:
            raise(e)
    
    def _yparse(self, filepath:str) -> dict:
        import yaml
        from beamngpy import Scenario
        with open(filepath, 'r') as file:
            obj = yaml.load(file, yaml.FullLoader)
        return obj