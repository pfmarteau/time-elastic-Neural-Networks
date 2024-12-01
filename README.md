# teNN, Time Elastic Neural Networks
### Implementation of teNN (Time Elastic Neural Networks) defined and experimented in \[[1](#1)\]. The teNN code itself is written in C and its Python wrapper uses the foreign ctype library.

The teNN architecture is based on the theory of time-elastic kernels and is mainly dedicated to the classification of univariate or multivariate time series. This atypical type of neural network architecture integrates a specific attention mechanism capable of handling temporal elasticity. In addition, teNN integrates a neuron pruning strategy (based on their inactivation) that, while dynamically reshaping the neural architecture itself, allows to drastically reduce the number of model parameters. teNN is efficiently trained using stochastic gradient descent using direct evaluation of categorical cross-entropy loss derivatives, thus avoiding the traditional backpropagation mechanism. Finally, it provides interpretable results and contributes to the explainable AI initiative.

<img src="figs/teNN-0.png" width="40%" height="40%"/><img src="figs/teNN.png" width="40%" height="40%"/>


## Requirements 
- gcc compiler
- python3.*
- matplotlib
- ScikitLearn

## To compile the c code
$ cd \<teNN install directory\>\
$ sh compile.sh


## Testing teNN on BME UCR dataset
$ python3  teNN_Benchmark.py --dataset BME --no_display --nu0 1e-6 --batch_size 64 --eta 1e-2 --nclusters 1 --lambda_At 1e-4 --lambda_Ac 1e-4 --corridor_radius 500

<span style="color:red">To stop the training process, type ^C (Ctrl c)</span>.


## [BME](https://www.timeseriesclassification.com/description.php?Dataset=BME) dataset
### Training loss
<img src="figs/BME_loss.png" width="40%" height="40%" style="display: block; margin: 0 auto" />

### Few samples for each category. The initial reference time series are presented in dotted red lines.

<img src="figs/BME_0.png" width="30%" height="30%"/><img src="figs/BME_1.png" width="30%" height="30%"/><img src="figs/BME_2.png" width="30%" height="30%"/>

### The activation matrices for each category
<img src="figs/BME_Activation_0.png" width="30%" height="30%"/><img src="figs/BME_Activation_1.png" width="30%" height="30%"/><img src="figs/BME_Activation_2.png" width="30%" height="30%"/>

### The attention matrices (vectors) for each category. The final class reference time series is given in blue dotted lines.
<img src="figs/BMEAttention_d0_0.png" width="30%" height="30%"/><img src="figs/BMEAttention_d0_1.png" width="30%" height="30%"/><img src="figs/BMEAttention_d0_2.png" width="30%" height="30%"/>

If you wish to reference teNN, please cite this draft paper:

<a name="1"></a>
[1] Marteau, P.F., Times Elastic Neural Networks, preprint, May 2024, arXiv:2405.17516, 
[ArXiv](https://arxiv.org/abs/2405.17516), 
[bibtex](https://github.com/pfmarteau/teNN/blob/main/bibtex/marteau2024.bib)

