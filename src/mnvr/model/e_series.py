import random
import tensorflow as tf

class Explorer_One(tf.keras.Model):
    """Explorative Model, Version 1
    
    A dummy model, of sorts.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(Explorer_One, self).__init__(*args, **kwargs)
        self.init_dir = (random.random() * 1.5) - 0.75
        self.init_throttle = 0.40
        
    def call(self, *args, **kwargs):
        y = [
                self.init_dir + ((random.random() * 0.25) - 0.125),     # Steering
                self.init_throttle + ((random.random() * 0.25) - 0.25), # Throttle
                (random.random() * 2) - 1.25                            # E-Brake
            ]
        
        return tf.convert_to_tensor(y)
