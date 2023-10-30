# A data-driven evolutionary transfer optimization for expensive problems in dynamic environments

**A data-driven evolutionary transfer optimization for expensive problems in dynamic environments**
[Ke Li]()\*, [Renzhi Chen]()\*, [Xin Yao]()\*
[[Paper]]() [[Supplementary]]()



## Overview

This repository contains Python implementation of the algorithm framework for Batched Data-Driven Evolutionary Multi-Objective Optimization Based on Manifold Interpolation.



## Code Structure

algorithms/ --- algorithms definitions

problems/ --- multi-objective problem definitions

revision/ -- patch for Gpy package

scripts/ --- scripts for batch experiments

 ├── build.sh --- complie the c lib for test problems

 ├── run.sh -- run the experiment 

main.py --- main execution file

## Requirements

- Python version: tested in Python 3.7.7
- Operating system: tested in Ubuntu 20.04



## Getting Started

### Basic usage

Run the main file with python with specified arguments:

```bash
python3.7 main.py --problem Movingpeak --n-var 6 
```

### Parallel experiment

Run the script file with bash, for example:

```bash
./run.sh
```



## Result

The optimization results are saved in txt format. They are stored under the folder:

```
output/data/{problem}/x{n}y{m}/{algo}-{exp-name}/{seed}/
```

## Citation

If you find our repository helpful to your research, please cite our paper:

```
@article{KeLi2023,
  author={Li, Ke and Chen, Renzhi and Yao, Xin},
  journal={IEEE Transactions on Evolutionary Computation}, 
  title={A Data-Driven Evolutionary Transfer Optimization for Expensive Problems in Dynamic Environments}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TEVC.2023.3307244}}

```

