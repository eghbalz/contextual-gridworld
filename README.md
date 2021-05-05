# Contextual-Gridworld
This repository hosts the Contextual Grid-world environment used for the experiments in

Hamid Eghbal-zadeh*, Florian Henkel* and Gerhard Widmer <br>
"[Learning to Infer Unseen Contexts in Causal Contextual Reinforcement Learning](https://openreview.net/pdf?id=gPZP5ha9LpX)" <br>
*Self-supervision for Reinforcement Learning (SSL-RL) Workshop ICLR, 2021, Vienna, Austria*



### Setup and Requirements
First, clone the project from github.
```
git clone https://github.com/eghbalz/contextual-gridworld.git
```

Navigate to the directory
```
cd contextual-gridworld
```

As a next step you need to install the required packages. If your are an anaconda user,
we  provide an [anaconda environment file](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
which can be installed as follows:
```
conda env create -f environment.yml
```


Activate your conda environment:
```
source activate contextual_gridworld
```

and install the *contextual_gridworld* package in develop mode:
```
python setup.py develop --user
```

To verify that everything is working, try to run `python manual_control.py` in the  `contextual_gridworld` directory. 
You can control the agent using the arrow keys `UP`, `LEFT` and `RIGHT`.
