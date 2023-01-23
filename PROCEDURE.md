# Training procedure
Useful if you would like to recreate the experiments that were done while writing the code.

The model architecture is static, what is changes between runs is the optimizer used and the penalty falloff method, both can be modified in trainer.py, under src/mnvr.

Train the model through 16 episodes and collect the data saved under data/reports/runs, perform your preferred analytical methods on the results and see if any interesting developments occured with the various configurations.

The model will inevitably fail, make sure you take a moment to enjoy the show :)