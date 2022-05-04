# Graph-Relational Domain Adaptation (GRDA)
This repo contains the code for our ICLR 2022 paper: "[Graph-Relational Domain Adaptation](wanghao.in/paper/ICLR22_GRDA.pdf)". We are still re-organizing the code for the camera-ready version.

## Beyond Domain Adaptation: Brief Introduction for GRDA
Essentially GRDA goes beyond current (categorical) domain adaptation regime and proposes the first approach to **adapt across graph-relational domains**. We introduce a new notion, dubbed "**domain graph**", that to encode domain adjacency, e.g., a graph of states in the US with each state as a domain and each edge indicating adjacency. Theoretical analysis shows that *at equilibrium, GRDA recovers classic domain adaptation when the graph is a clique, and achieves non-trivial alignment for other types of graphs*. See the following example (black nodes as source domains and white nodes as target domains).

<p align="center">
<img src="fig/GRDA-domain-graph-US.png" alt="" data-canonical-src="fig/GRDA-domain-graph-US.png" width="95%"/>
</p>

## Installation
    pip install -r requirements.txt

## How to train
    python main.py

## Visualization
We use visdom to visualize. We assume the code is run on a remote gpu machine.

### Change config
Find the config in "config" folder. Choose the config you need and Set "opt.use_visdom" to "True".

### Start visdom server on gpu machine
    python -m visdom.server -p 2000
Now connect your computer with the gpu server and forward the port 2000 to your local computer. You can now go to:
    http://localhost:2000 (Your local address)
to see the visualization during training.

## Theory (Informal)
- Traditional DA is equivalent to using our GRDA with a fully-connected graph (i.e., a clique).
- D and E converge if and only if <img src="http://www.sciweavers.org/tex2img.php?eq=%20%5Cfrac%7Ba%7D%7Bb%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt=" \frac{a}{b} " width="17" height="40" /> .
- The global optimum of the two-player game between E and D matches the three-player game between E, D, and F.

<!-- <img src="https://latex.codecogs.com/svg.image?E_{i,j}[A_{i,j}|e_i,e_j]&space;=&space;E_{i,j}[A_{i,j}]" /> -->

## Reference
If you find our work useful in your research, please consider citing:
```
@inproceedings{GRDA,
  title={Graph-Relational Domain Adaptation},
  author={Xu, Zihao and He, Hao and Lee, Guang-He and Wang, Yuyang and Wang, Hao},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```
