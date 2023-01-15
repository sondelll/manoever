import tensorflow as tf
from PIL import Image
from ..simulation import SimulationResult

class JsonRunReport:
    """Writes a report
    """
    def __init__(
        self,
        run_result:SimulationResult,
        agent:tf.keras.Model,
        predictor:tf.keras.Model,
        report_dir:str = "./data/reports/runs"
    ) -> None:
        self.rep_dir = report_dir
        self.agent = agent.__class__.__name__
        self.predictor = predictor.__class__.__name__
        
        slim_result = run_result.to_slim_dict()
        slim_result['damage'] = slim_result['damage']['damage']
        slim_result['agent_model'] = self.agent
        slim_result['predictor_model'] = self.predictor
        
        self.data = slim_result
        
    def save(self):
        from time import time
        name = f"{int(time())}_{self.agent}x{self.predictor}.json"
        save_path = f"{self.rep_dir}/{name}"
        json_write(save_path, self.data)


    
def json_write(path:str, data:dict):
        import json
        try:
            with open(path, 'w') as out_file:
                content = json.dumps(data, indent=4)
                out_file.write(content)
                out_file.flush()
                out_file.close()
            
        except Exception as e:
            print(e)

   
def dump_to_png(raw_depth:Image.Image, suffix) -> None:
    try:
        processed_depth = tf.convert_to_tensor(
            list(raw_depth.getdata())
        )
        
        image_data = tf.reshape(
            processed_depth,
            (32,256,1)
        )
        
        u8_img = tf.cast(image_data, tf.uint8)
        as_png = tf.image.encode_png(u8_img)
        
        tf.io.write_file(
            f"./camera_{str(suffix)}_debugfile.png",
            as_png
        )
    except Exception as e:
        print(e)