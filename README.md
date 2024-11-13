# teNN, Time Elastic Neural Networks
### Implementation of teNN (Time Elastic Neural Networks) defined and experimented in [[1](#1)]. The teNN code itself is written in C and its Python wrapper uses the foreign ctype library.

## Requirements 
- gcc compiler
- python3.*
- matplotlib
- ScikitLearn

## To compile the c code
$ cd \<teNN install directory\>\
$ sh compile.sh


## Testing teNN on BME UCR dataset
$ python3  teNN_Benchmark.py --dataset BME --no_display --nu0 1e-6 --batch_size 0 --eta 1e-2 --nclusters 1 --lambda_At 1e-4 --lambda_Ac 1e-4 --corridor_radius 500

## BME dataset
### Training loss
<img src="figs/BME_loss.png" width="40%" height="40%" style="display: block; margin: 0 auto" />

### Few samples for each category. The initial reference time series are presented in dotted red lines.

<img src="figs/BME_0.png" width="30%" height="30%"/><img src="figs/BME_1.png" width="30%" height="30%"/><img src="figs/BME_2.png" width="30%" height="30%"/>

### The activation matrices for each category
<img src="figs/BME_Activation_0.png" width="30%" height="30%"/><img src="figs/BME_Activation_1.png" width="30%" height="30%"/><img src="figs/BME_Activation_2.png" width="30%" height="30%"/>

### The attention matrices (vectors) for each category. The final class reference time series is given in blue dotted lines.
<img src="figs/BMEAttention_d0_0.png" width="30%" height="30%"/><img src="figs/BMEAttention_d0_1.png" width="30%" height="30%"/><img src="figs/BMEAttention_d0_2.png" width="30%" height="30%"/>

Please cite this article if you wish to reference teNN:

<a name="#1">
   [1] </a> 
Marteau, P.F., Times Elastic Neural Networks, preprint, [ArXiv May 2024](https://arxiv.org/abs/2405.17516), 
[bibtex](./bibtex/marteau2024.bib)

