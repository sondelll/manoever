from dataclasses import dataclass, field, asdict

import numpy as np
import json
import tensorflow as tf


class TensorCompat:
    """Tensorflow compatible, for subclassing.
    
    Standard way of getting data out of class and into tensor.
    """
    def to_tensor(self) -> tf.Tensor:
        """Get a model-friendly tensor representing the dataclass.

        Raises:
            NotImplementedError: if missing in subclass

        Returns:
            tf.Tensor: The data as a tensor.
        """
        raise NotImplementedError("Required method")

    def to_dict(self):
        """Dataclass to plain dict object.

        Returns:
            dict: Data from field(s) of data class.
        """
        return asdict(self)

   
@dataclass
class NGVehicleSense(TensorCompat):
    """Dataclass for numerical vehicle sensor data
    """
    airspeed:float = field(init=True, default=None)
    angular_acceleration:tuple[float, float, float] = field(init=True, default=None)
    angular_velocity:tuple[float, float, float] = field(init=True, default=None)
    powertrain_speed:float = field(init=True, default=None)

    def to_json_file(self, pathname:str = "./data/percepted.json", prettyprint:bool = True):
        """Dump data to json, good for debugging.
        
        Note: Not very storage efficient.

        Args:
            pathname (str, optional): _description_. Defaults to "./data/percepted.json".
            prettyprint (bool, optional): Use indentation etc., for readability. Defaults to True.
        """
        ind = 4 if prettyprint else None
        
        j_str = json.dumps(self.to_dict(), sort_keys=True, indent=ind)
        with open(pathname, 'w') as f:
            f.write(j_str)
            f.flush()
            f.close()
            
    def to_tensor(self) -> tf.Tensor:
        float_vals = []
        iterables = []
        d = self.to_dict()
        for key, val in d.items():
            if isinstance(val, float): # We just need the values then
                float_vals.append(val/100) # Scale down by factor of 100
            elif isinstance(val, list): # This is trickier
                t = tf.convert_to_tensor(val, dtype=tf.float32)
                #t = tf.divide(t, 100) # Scale down by factor of 100
                iterables.append(t)
                
        iters_tensor = iterables.pop(0) # Get first tensor of iterables
        for it in iterables: # Concat the rest in a loop
            iters_tensor = tf.concat([iters_tensor, it], 0)
        
        float_tensor = tf.convert_to_tensor(float_vals, dtype=tf.float32)
        tensor = tf.concat([iters_tensor, float_tensor], 0) # Join w floats.
        return tensor

@dataclass
class NGSight(TensorCompat):
    raw_depth:list = field(init=True, default=None)

    def to_png(self, out_dir:str = "./data", run_id:str = "debug", step_n:int = 0):
        """Saves depth image to png, mainly for debugging.

        Args:
            out_dir (str, optional): Output directory (no trailing slash!). Defaults to "./data".
            run_id (str, optional): Identifier name for simulation run. Defaults to "debug".
            step_n (int, optional): Number of steps into simulation run. Defaults to 0.
        
        Example:
            sight = NGSight(data_in)
            
            sight.to_png(run_id="FastWroom", step_n=17)
            
                -> Writes data to image file at ./data/depth_FastWroom_17.png
        """
        import tensorflow as tf
        savename = f"{out_dir}/depth_{run_id}-{step_n}.png"
        
        try:
            depth_data = tf.convert_to_tensor(list(self.raw_depth.getdata()))
            with_image_shape = tf.reshape(depth_data(256, 32))
            to_uint8_scale = tf.multiply(with_image_shape, 255)
            with_channel_dim = tf.expand_dims(to_uint8_scale, -1)
            as_uint8 = tf.cast(with_channel_dim, tf.uint8)
            
            encoded = tf.image.encode_png(image=as_uint8)
            tf.io.write_file(savename, encoded)
        except Exception as e:
            print(e)
            
    def to_tensor(self) -> tf.Tensor:
        depth_data = tf.convert_to_tensor(list(self.raw_depth.getdata()))
        depth_floatrange = tf.divide(depth_data, 255.0)
        return depth_floatrange

@dataclass
class NGTermination:
    sight:NGSight = field(init=True, default=None)
    sense:NGVehicleSense = field(init=True, default=None)
    damage:dict = field(init=True, default=None)
    