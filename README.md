## Setting up the environment

**conda** should be used to get a python environment with FEniCS and pytorch. Create with

```
	conda env create -f  fenicsandpytorch.yml
```

and activate with

```
	conda activate fenicsandpytorch
```


### Create a Notebook

Navigate to the corresponding directory and start the Notebook server using 

```
  jupyter notebook --port=8888
```

The server is now accessible at ``localhost:8888``.

In a computing server / cluster environment invoke the command 

```
  jupyter notebook --no-browser --port=8888
```

and enable port forwarding. To do so, open a *local* terminal and execute

```
  ssh -N -L localhost:8888:localhost:8888 username@server
```