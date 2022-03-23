# Graph-Relational Domain Adaptation
This repo contains the code for our ICLR 2022 paper: "Graph-Relational Domain Adaptation". We are still re-organizing the code for the camera-ready version.

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
