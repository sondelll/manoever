import tensorflow as tf


class SimulationResult:
    """Class abstraction of simulation result.
    """
    samples:tuple
    states:list[tf.Tensor]
    actions:list[tf.Tensor]
    
    damage:dict
    electrical_data:dict
    distance_traveled:float
    
    def __init__(self, samples:list[tuple[tf.Tensor, tf.Tensor]], simulation_info:tuple[dict, dict, float]) -> None:
        self.samples = samples
        self.states = [sample[0] for sample in samples]
        self.actions = [sample[1] for sample in samples]
        
        self.damage = simulation_info[0]
        self.electrical_data = simulation_info[1]
        self.distance_traveled = simulation_info[2]


    def penalty(
        self,
        damage_factor:float = 7e-2,
        distance_factor:float = 1e-1,
        penalty_multiplier:float = 0.1
    ) -> float:
        import random
        pen = 0.0
        try:
            pen += float(self.damage['damage']) * (random.random() + 0.95) # Randomized multiplier to keep it from sneaking in damage oriented strats
        except:
            print("An error occurred during damage penalty calculation.")
            
        pen *= damage_factor
        
        try:
            if self.distance_traveled < 500: # reasonable threshold for "not getting anywhere" penalty(?)
                pen += (500 - self.distance_traveled) * distance_factor
        except:
            print("Failed to find the distance traveled by the car.")

        return pen * penalty_multiplier # It causes way too large corrections otherwise


    def to_dict(self) -> dict:
        """Get result as dict, all inclusive.

        Returns:
            dict: The simulation results
        """
        return {
            "penalty":self.penalty(),
            "states":self.states,
            "actions":self.actions,
            "damage":self.damage,
            "distance":self.distance_traveled
            }

    
    def to_slim_dict(self) -> dict:
        """Get results as dict, without states and actions.

        Returns:
            dict: The simulation results
        """
        return {
            "penalty":self.penalty(),
            "damage":self.damage,
            "distance":self.distance_traveled
            }
