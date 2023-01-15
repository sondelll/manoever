import tensorflow as tf


class P6(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(P6, self).__init__(*args, **kwargs)
        
        self.d1 = tf.keras.layers.Dense(units=4096, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.5)
        self.d2 = tf.keras.layers.Dense(units=2048, activation='relu')
        self.d3 = tf.keras.layers.Dense(units=1024, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d4 = tf.keras.layers.Dense(units=1)
        
    @tf.function(reduce_retracing=True)
    def call(self, inputs, training=None, mask=None):
        y = self.d1(inputs)
        if training:
            y = self.drop(y)
        y = self.d2(y)
        y = self.d3(y)
        y = self.flatten(y)
        y = self.d4(y)
        return y
    
class P7(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(P7, self).__init__(*args, **kwargs)
        
        self.d1 = tf.keras.layers.Dense(units=1024, activation='elu')
        
        self.drop = tf.keras.layers.Dropout(0.2)
        
        self.d2 = tf.keras.layers.Dense(units=512, activation='elu')
        self.d3 = tf.keras.layers.Dense(units=256, activation='relu')
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.d4 = tf.keras.layers.Dense(units=1)
        
    @tf.function(reduce_retracing=True)
    def call(self, inputs, training=None, mask=None):
        y = self.d1(inputs)
        if training:
            y = self.drop(y)
        y = self.d2(y)
        y = self.d3(y)
        y = self.flatten(y)
        y = self.d4(y)
        return y
