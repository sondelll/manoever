import tensorflow as tf

from ..simulation import ScenarioParseV1, AgentNGSimulation, ExpertNGSimulation, SimulationResult


class ManoeverRunner:
    """Abstract simulation run handler.
    """
    def __init__(self, scenario = None) -> None:
        self.scenario = scenario
    
    
    def preload_scenario(self, filepath:str) -> None:
        """Preload scenario, overrides scenario passed into constructor if any.

        Args:
            filepath (str): Path to *.sc.yaml scenario file.
        """
        sp = ScenarioParseV1()
        self.scenario, _ = sp.from_sc_file(filepath)
    
    
    def run_simulator(
        self,
        agent:tf.keras.Model = None
    ) -> SimulationResult:
        """Do a simulation run.

        Args:
            agent (tf.keras.Model): Model that outputs three float values.
            scenario (beamngpy.Scenario, optional): Scenario, required if no preload has been executed. Defaults to None.

        Returns:
            SimulationResult: Results from simulation.
        """
        self.predictor_ret = False # Catch-all in case someone misses to disable
        if agent is not None:
            sim = AgentNGSimulation(self.scenario)
            samples, sim_info = sim.run(agent, truncate_seconds=10.0)
            del sim
        else:
            sim = ExpertNGSimulation(self.scenario)
            samples, sim_info = sim.run(truncate_seconds=10.0)
            del sim

        return SimulationResult(samples, sim_info)
    

    