import tensorflow as tf

from ..model.e_series import Explorer_One
from .runner import ManoeverRunner
from ..simulation.scenarioparse import ScenarioParseV1
from ..optimizer import Pang
    
class AgentTrainer:
    def __init__(
        self,
        agent:tf.keras.Model
    ) -> None:
        self.agent = agent
        self.predictor = agent.predictor_model

    def train_from_results(
        self,
        results
    ):
        history = self.fit_agent(results.samples)
        
        return history
    
    def fit_agent(
        self,
        samples
    ) -> tf.keras.callbacks.History:
        print("AGENT TRAIN START -----|")
        self.agent.predictor_model = self.predictor
        
        self.agent.compile(
            #optimizer=Pang(learning_rate=1e-4, amp=1e-3), # Picard
            #optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3), # Kirk First 1e-4, later 1e-3
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Janeway
            loss=tf.keras.losses.MeanSquaredError()
        )

        self.agent.predictor_model.trainable = False # To make sure it doesn't train during agent fitting
        self.agent.predictor_ret = True
        
        x, y = self._agent_preprocessing(samples)
        
        history = self.agent.fit(
            x, y,
            epochs=1,
            verbose=1,
            batch_size=x.shape[0]
        )
        
        if self.agent.predictor_model is not None:
            self.agent.predictor_model.trainable = True # Restoring trainability
        self.agent.predictor_ret = False
        
        self.agent.save(f"./data/saved_models/{self.agent.__class__.__name__}", include_optimizer=False)
        
        return history
        
    def _agent_preprocessing(
        self,
        samples:list[tuple[tf.Tensor, tf.Tensor]],
        with_jitter:bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        from .jitter import extend_by_jitter
        processed_samples = []
        state_samples = [sample[0] for sample in samples]
        if with_jitter:
            processed_samples = extend_by_jitter(state_samples, extension=2)
        else:
            processed_samples = tf.squeeze(state_samples, 1)
        
        x_data = tf.convert_to_tensor(processed_samples)
        y_data = tf.fill(
            dims=((len(processed_samples))),
            value=0.001
        )
        
        return x_data, y_data
    

class PredictorTrainer:
    def __init__(self, predictor:tf.keras.Model) -> None:
        self.predictor = predictor
        self.agent = Explorer_One()
        
    def train_with_explorer(
        self,
        sc_filepath:str = "./scenarios/1A.sc.yaml"
    )-> tf.keras.callbacks.History:
        
        sp = ScenarioParseV1()
        # Below scenario must be redefined before every run
        scenario, (_pos, _rot) = sp.from_sc_file(sc_filepath)
        
        runner = ManoeverRunner(scenario)
        sim_result = runner.run_simulator(self.agent)
        self.fit_predictor(sim_result)

    def fit_predictor(self, results):
        x, y = self._prediction_preprocessing(results.samples, results.penalty())
        
        self.predictor.compile(
            #optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5), # Kirk / Picard, LR1e-5
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Janeway
            loss=tf.keras.losses.MeanSquaredError()
        )
        
        self.predictor.trainable = True
        
        history = self.predictor.fit(
            x, y,
            epochs=4,
            verbose=1,
            batch_size=x.shape[0]
        )
        
        self.predictor.save(f"./data/saved_models/{self.predictor.__class__.__name__}")
        
        return history
    
    def train_from_results(self, results):
        print("PREDICTOR TRAIN START -----|")
        history = self.fit_predictor(results)
        
        return history
        
    def insert_custom_agent(self, agent:tf.keras.Model):
        self.agent = agent
    
    def _prediction_preprocessing(
        self,
        samples:tuple,
        penalty:float
    ) -> tuple[tf.Tensor, tf.Tensor]:
        processed_samples = []
        
        for sample in samples:
            state = tf.squeeze(sample[0])
            action = tf.squeeze(sample[1])
            combined = tf.concat([state, action], 0)
            processed = tf.expand_dims(combined, 0)
            
            processed_samples.append(processed)
        
        x_data = tf.convert_to_tensor([processed_samples])
        
        # y_data = get_penalty_falloff(len(processed_samples), penalty) # Falloff
        y_data = get_penalty_falloff(len(processed_samples), penalty, from_term=False) # Inverted Falloff
        # y_data = get_penalty_flat(len(processed_samples), penalty) # Flat penalty
        
        x_data = tf.squeeze(x_data)
        y_data = tf.squeeze(y_data)
        return x_data, y_data

def get_penalty_falloff(
    length:int,
    penalty:float,
    from_term:bool = True
):
    _y_data = []
    for step in range(length):
            _tdf = 0.8**(step + 1)
            # This is SPICY logic, but at least it's interesting.
            temporal_distance_factor = _tdf if _tdf > 0.01 else 0.01
            _y_data.append(penalty * temporal_distance_factor)
            
            if from_term:
                _y_data.reverse()
    return tf.convert_to_tensor(_y_data)

def get_penalty_flat(length:int, penalty:float):
    return tf.fill((length), value=penalty)