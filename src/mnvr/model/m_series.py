import tensorflow as tf

import numpy as np

import random

class MSeriesModel(tf.keras.Model): # TODO: Add relevant base fields
    predictor_model:tf.keras.Model
    predictor_ret:bool
  
class M31(MSeriesModel):
    def __init__(self, predictor:tf.keras.Model, visual_dims:tuple[int, int], *args, **kwargs):
        super(M31, self).__init__(*args, **kwargs)
        self.predictor_model = predictor
        self.predictor_ret = False
        self.visual_dims = visual_dims
        self.visual_size = visual_dims[0] * visual_dims[1]
        
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(9,9),
            strides=(2,2), activation='elu'
        )
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(5,5),
            strides=(2,2), activation='elu'
        )
        self.conv_3 = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=(3,3),
            strides=(2,2), activation='elu'
        )
        self.flatten = tf.keras.layers.Flatten()
        
        self.steering = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=256, activation='elu'),
                tf.keras.layers.Dense(
                    units=128,
                    activation='elu'
                ),
                tf.keras.layers.Dense(units=1)
            ],
            name='steering'
        )
        
        self.eb = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=16, activation='elu'),
                tf.keras.layers.Dense(units=2, activation='elu'),
                tf.keras.layers.Dense(units=1)
            ],
            name='eBrake'
        )
        
        self.longitude = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=512, activation='elu'), # Needs a kick-start
                tf.keras.layers.Dense(units=32, activation='elu'),
                tf.keras.layers.Dense(units=1, activation='sigmoid')
            ],
            name='longitude'
        )
        
        self.visual_reshape = tf.keras.layers.Reshape((self.visual_dims[0], self.visual_dims[1], 1))

    @tf.function 
    def call(
        self,
        inputs:tf.Tensor,
        training:bool = False
    ):
        #inputs = tf.squeeze(inputs)
        if training:
            inputs = tf.keras.layers.Dropout(0.25)(inputs)
        # Split input into components
        x_visual = tf.map_fn(self.unstack_visual, inputs)
        x_data = tf.map_fn(self.unstack_data, inputs)
        
        # Camera input to actual convolvable shape
        x_visual = self.visual_reshape(x_visual)
        
        # Because of beamNG sometimes causing float32's to be passed in, for whatever reason
        x_visual = tf.cast(x_visual, tf.float32)
        x_data = tf.cast(x_data, tf.float32)
        
        y_visual = self._call_visual_convolution(x_visual)
        
        y = tf.concat([y_visual, x_data], -1, name='recombine')
        
        steer:tf.Tensor = self.steering(y)
        ltd:tf.Tensor = self.longitude(y)
        e_brake = self.eb(y)
        
        control_tensor = tf.concat([steer, ltd, e_brake], -1, name='ctrl_concat') #tf.squeeze()
        try:
            using_predictor = self._ppred_call(inputs, control_tensor)
            if self.predictor_ret: # Hold on now, we're taking a detour
                return using_predictor
        except Exception as e:
            print("Predictor error: ", e)
        
        return control_tensor            
    
    @tf.function(reduce_retracing=True)
    def unstack_visual(self, t:tf.Tensor) -> tf.Tensor:
        return t[:self.visual_size]
    
    @tf.function(reduce_retracing=True)
    def unstack_data(self, t:tf.Tensor) -> tf.Tensor:
        return t[self.visual_size:]
    
    @tf.function
    def _call_visual_convolution(self, x):
        y = self.conv_1(x)
        y = self.conv_2(y)
        y = self.conv_3(y)
        y = self.flatten(y)
        return y
    
    @tf.function    
    def _ppred_call(self, input_state, control_tensor):
        if self.predictor_model is None:
                raise ValueError("In training, having a predictor is required.")
        
        # Adding extra dim to match (1, size) shape
        #input_state = tf.expand_dims(input_state, 0)
        #control_tensor = tf.expand_dims(control_tensor, 0)
        prediction = self.predictor_model(
            tf.concat([
                input_state,
                control_tensor
                ],
                axis=1,
                name='prediction_prep_concat'
            )
        )
        # Here we pass back the data ran through both agent and predictor network,
        # to avoid a gradient-breaking gap
        return prediction
    
class M32(MSeriesModel):
    def __init__(self, predictor:tf.keras.Model, visual_dims:tuple[int, int], *args, **kwargs):
        super(M32, self).__init__(*args, **kwargs)
        self.predictor_model = predictor
        self.predictor_ret = False
        self.visual_dims = visual_dims
        self.visual_size = visual_dims[0] * visual_dims[1]
        
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(7,7),
            activation='elu'
        )
        
        self.pool1 = tf.keras.layers.MaxPool2D((4,2), padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(5,5),
            activation='elu'
        )
        self.pool2 = tf.keras.layers.MaxPool2D((4,2), padding='same')
        
        self.conv_3 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3,3),
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1234),
            activation='elu'
        )
        self.pool3 = tf.keras.layers.MaxPool2D((2,2), padding='same')
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.drop = tf.keras.layers.Dropout(0.1)
        self.steering = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    units=1024, activation='elu',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=2345)
                ),
                tf.keras.layers.Dense(units=1)
            ],
            name='steering'
        )
        
        self.eb = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    units=32, activation='elu',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=3456)
                ),
                tf.keras.layers.Dense(units=1)
            ],
            name='eBrake'
        )
        
        self.longitude = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    units=1024,
                    activation='elu',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=5678)
                ),
                tf.keras.layers.Dense(units=1)
            ],
            name='longitude'
        )
        
        self.visual_reshape = tf.keras.layers.Reshape((self.visual_dims[0], self.visual_dims[1], 1))

    @tf.function 
    def call(
        self,
        inputs:tf.Tensor,
        training:bool = False
    ):
        #inputs = tf.squeeze(inputs)
        if training:
            inputs = tf.keras.layers.Dropout(0.1)(inputs)
        # Split input into components
        x_visual = tf.map_fn(self.unstack_visual, inputs)
        x_data = tf.map_fn(self.unstack_data, inputs)
        
        # Camera input to actual convolvable shape
        x_visual = self.visual_reshape(x_visual)
        
        # Because of beamNG sometimes causing float64's to be passed in, for whatever reason
        x_visual = tf.cast(x_visual, tf.float32)
        x_data = tf.cast(x_data, tf.float32)
        
        y_visual = self._call_visual_convolution(x_visual)
        
        y = tf.concat([y_visual, x_data], -1, name='recombine')
        if training:
            y = self.drop(y)
        steer:tf.Tensor = self.steering(y)
        ltd:tf.Tensor = self.longitude(y)
        e_brake = self.eb(y)
        
        control_tensor = tf.concat([steer, ltd, e_brake], -1, name='ctrl_concat') #tf.squeeze()
        try:
            using_predictor = self._ppred_call(inputs, control_tensor)
            if self.predictor_ret: # Hold on now, we're taking a detour
                return using_predictor
        except Exception as e:
            print("Predictor error: ", e)
        
        return control_tensor            
    
    @tf.function(reduce_retracing=True)
    def unstack_visual(self, t:tf.Tensor) -> tf.Tensor:
        return t[:self.visual_size]
    
    @tf.function(reduce_retracing=True)
    def unstack_data(self, t:tf.Tensor) -> tf.Tensor:
        return t[self.visual_size:]
    
    @tf.function
    def _call_visual_convolution(self, x):
        y = self.conv_1(x)
        y = self.pool1(y)
        y = self.conv_2(y)
        y = self.pool2(y)
        y = self.conv_3(y)
        y = self.pool3(y)
        y = self.flatten(y)
        return y
    
    @tf.function    
    def _ppred_call(self, input_state, control_tensor):
        if self.predictor_model is None:
                raise ValueError("In training, having a predictor is required.")
        
        # Adding extra dim to match (1, size) shape
        #input_state = tf.expand_dims(input_state, 0)
        #control_tensor = tf.expand_dims(control_tensor, 0)
        prediction = self.predictor_model(
            tf.concat([
                input_state,
                control_tensor
                ],
                axis=1,
                name='prediction_prep_concat'
            )
        )
        # Here we pass back the data ran through both agent and predictor network,
        # to avoid a gradient-breaking gap
        return prediction