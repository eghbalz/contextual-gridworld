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


## Citation

```
@InProceedings{pmlr-v148-eghbal-zadeh21a,
  title     =  {Context-Adaptive Reinforcement Learning using Unsupervised Learning of Context Variables},
  author    =  {Eghbal-zadeh, Hamid and Henkel, Florian and Widmer, Gerhard},
  booktitle =  {NeurIPS 2020 Workshop on Pre-registration in Machine Learning},
  pages     =  {236--254},
  year      =  {2021},
  editor    =  {Bertinetto, Luca and Henriques, João F. and Albanie, Samuel and Paganini, Michela and Varol, Gül},
  volume    =  {148},
  series    =  {Proceedings of Machine Learning Research},
  month     =  {11 Dec},
  publisher =  {PMLR},
  pdf       =  {http://proceedings.mlr.press/v148/eghbal-zadeh21a/eghbal-zadeh21a.pdf},
  url       =  {http://proceedings.mlr.press/v148/eghbal-zadeh21a.html}
}

@inproceedings{eghbal2021learning,
  title={Learning to Infer Unseen Contexts in Causal Contextual Reinforcement Learning},
  author={Eghbal-zadeh, Hamid and Henkel, Florian and Widmer, Gerhard},
  booktitle={Self-supervision for Reinforcement Learning (SSL-RL),  ICLR 2021 Workshop},
  year={2021}
}
```
