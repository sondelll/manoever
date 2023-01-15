import numpy as np
import tensorflow as tf
import random

def extend_by_jitter(samples:list[tf.Tensor], probability:float = 0.25, max_mag:float = 5e-2, extension:int = 3):
    out = []
    for sample in samples:
        out.append(tf.squeeze(sample))
        for n in range(extension):
            jittered = _jitter_sample(sample, probability, max_mag)
            out.append(tf.squeeze(jittered))
    return out


def _jitter_sample(sample:tf.Tensor, p:float, m:float):
    effector:np.ndarray = np.ones(shape=(tf.size(sample)), dtype=np.float32)
    
    for n in range(tf.size(sample)):
        if random.random() < p:
            effector[n] *= 1 + (random.random() - 0.5) * m
            
    effector = tf.reshape(effector, sample.shape)
    result = tf.multiply(sample, effector)
    return result
