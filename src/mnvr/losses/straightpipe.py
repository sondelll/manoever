import tensorflow as tf

class StraightPipe(tf.losses.Loss):
    def call(self, y_true, y_pred):
        return y_pred