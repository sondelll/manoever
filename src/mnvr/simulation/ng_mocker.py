import tensorflow as tf

from ..io.yaml_read import MnvrConfig

class MockNGV1:
    def run(self, n_samples:int = 24) -> tuple[
        list[tuple[tf.Tensor, tf.Tensor]],
        tuple[dict, dict, float]
    ]:
        samples = self._samples(n_samples)
        damage = {"damage":150}
        elec_data = {
            
        }
    
    
    
    def _samples(self, length:int):
        sensor_data = self._sensor_outputs(length)
        agent_responses = self._agent_responses(length)
        samples = zip(sensor_data, agent_responses)
        return samples
        
        
    def _agent_responses(self, length:int):
        data = []
        for _n in range(length):
            action_data = tf.random.truncated_normal(
                shape=(1, 3),
                mean=0.5,
                stddev=0.25
            )
            
            data.append(action_data)
        return data
    
    def _sensor_outputs(self, length:int) -> list[tf.Tensor]:
        data = []
        for _n in range(length):
            camera_data = tf.random.truncated_normal(
                shape=(1, 8192),
                mean=0.5,
                stddev=0.25
            )
            
            sensor_data = tf.random.truncated_normal(
                shape=(1, 12),
                mean=0.5,
                stddev=0.25
            )
            
            full_sensor_data = tf.concat(
                [camera_data, sensor_data],
                axis=-1
            )
            
            data.append(full_sensor_data)
        return data