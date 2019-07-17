# TODO: write user-frienldy readme.md;

dependencies : 

- tensorflow-gpu
- keras
- snntoolbox
- numpy


- To train new ANN Models on mnist or cifar-10, specify parameters in config.py.
  To alter the network architecture, add new parameters in ordered dict (stick to naming convention in config.py). Only conv, dropout, flatten, and dense layers are supperted yet.
In a console session, run in ./scr/ :
	>>> python main.py


- See models/ANN/models_summary.txt for more details on the network architecture 


- to start start a new simulation of a spiking neural network, use one of the config files in ./models/ANN/<dataset>/configs and run :
	>>> snntoolbox -t <config_file>




