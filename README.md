# Defrise and Clack Neural Network
[![arXiv](https://img.shields.io/badge/arXiv-2403.00426-b31b1b.svg)](https://arxiv.org/abs/2403.00426)

PyTorch implementation of the paper ["Deep Learning Computed Tomography based on the Defrise and Clack Algorithm"](https://arxiv.org/abs/2403.00426). This repository includes the code for a data-driven methodology for reconstructing CBCT projections for a given orbit.

## Requirements

The Defrise and Clack Neural Network code is developed using Python 3.11, PyTorch 2.1.1 and PyTorch-lightning 2.1.2. To ensure compatibility, please install the necessary packages using the following commands to create and activate a conda environment:

```bash
conda env create -f environment.yml
conda activate Defrise_and_Clack_Neural_Network
```


## Data
The simulation data set used for training can be generated by executing simulated data/data_gen_2.py.

## checkpoints
The pre-trained model is saved under the path checkpoints/checkpoint.ckpt.

## Code Structure

This repository is organized as follows:

- `simulated data/data_gen_2.py`: This script is responsible for generating the dataset.

- `dataset.py`: This script is responsible for handling the dataset.

- `DandCReconstrucion.py`: Contains the implementation of the Defrise and Clack Neural Network.

- `intermediateFunction.py`: Calculation for the Grangeat's intermediate function.

- `weight.py`: Defines the weight layers required in the reconstruction process.

- `train.py`: Execute it to train the neural network.

- `reference.py`: Execute it to test the neural network.


## Citation

```
@article{ye2024deep,
  title={Deep Learning Computed Tomography based on the Defrise and Clack Algorithm},
  author={Ye, Chengze and Schneider, Linda-Sophie and Sun, Yipeng and Maier, Andreas},
  journal={arXiv preprint arXiv:2403.00426},
  year={2024}
}
```
## Acknowledgments

- Thanks to [sypsyp97](https://github.com/sypsyp97) for his [Eagle_Loss](https://github.com/sypsyp97/Eagle_Loss), which was a great reference in building this application.

