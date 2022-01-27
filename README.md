# Graph-Relational Domain Adaptation
This is the code for Graph-Relational Domain Adaptation (ICLR 2022). We are still re-organizing the code for camera-ready version.

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
    http://localhost:xxx(Your local address)
to see the visualization during training.
