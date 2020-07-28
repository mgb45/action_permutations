# Soma puzzle experiments

## Model training and Data generation

Soma puzzle disassembly experiments use a Coppeliasim environment. The notebook [soma_sim.ipynb](./soma_sim.ipynb) generates data for testing (images of the 240 soma puzzle configurations considered here). This notebook extracts soma puzzle solutions (obtained using [polyform puzzler](http://puzzler.sourceforge.net/)) from this [text file](./soma_cube.txt).

Manually definied ground truth extraction orders are in this [file](./data/extraction_order.txt).

- The notebook [soma_sim_model_trainer.ipynb](soma_sim_model_trainer.ipynb) trains an action sequencing model using a Sinkhorn network
- The notebook [soma_sim_model_trainer_tcn.ipynb](soma_sim_model_trainer_tcn.ipynb) trains an action sequencing model using a TCN

## Model testing

Test scripts are in the scripts folder. 
- [soma_sim_model_trainer_sink.py](./scripts/soma_sim_model_trainer_sink.py) runs planning tests using the Sinkhorn predictions
- [soma_sim_model_trainer_tcn.py](./scripts/soma_sim_model_trainer_sink.py) runs planning tests using the TCN predictions

## Evaluation

Notebook for results plotting are in the data folder.
