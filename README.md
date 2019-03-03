# Collaborative-Filtering-RBM


### How to run

To run the project execute

```sh
$ git clone https://github.com/vikasmahato/Collaborative-Filtering-RBM.git
$ cd Collaborative-Filtering-RBM
$ make run
```

The project runs all tests defined in the Makefile

```sh
$ THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP1 pythonw cfrbm/nosparse_item_based.py experiment_descriptions/100k_u1_ibased.json
```

There are various experiments described in ```experiment_descriptions``` and on running the project, the experiments are executed and their results are stored in ```experiments```. It may take ***more than a day*** to run all experiments.

To speed up the process, you can utilize the gpu of your machine by replacing 

```device=cpu```
with
```
device=gpu
```
in Makefile and Makefile.100k if you have a gpu configured on your device

On Mac ```python2``` is installed and ```python3``` conflicts with the standard installation.
Hence ```pythonw``` is installed with conda.

**For running on Linux and Windows replace ```pythonw``` with ```python``` in Makefile and Makefile.100k**

### Libraries

- numpy
- sciPy
- pandas
- matplotlib
- theano
- scikit learn