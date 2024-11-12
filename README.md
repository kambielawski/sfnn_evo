# Evolution of Structurally Flexible NNs

**evo.py** -  Evolutionary algorithms; fitness functions; mutation and crossover operations

**sffn.py** - Torch implementation of Structurally Flexible NN

**rl.py** - Implementation of RL environment and evaluation of a SFNN

## Running an experiment

1. Create an experiment file

Example:

```
{
    "ea": "HillClimber",
    "n_runs": 1,
    "evo_parameters": {
        "population_size": 10,
        "n_generations": 1000,
        "mutation_rate": 0.01,
        "neuron_size": 10,
        "gru_size": 10,
        "hidden_layer_size": 10,
        "lr": 0.1,
        "ticks": 2
    }
}
```

2. Run `run_exp.py` and pass in the experiment file as a CLI parameter

Example:
```
python3 run_exp.py exp_file_example.json
```
