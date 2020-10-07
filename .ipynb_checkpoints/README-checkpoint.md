# deep-learning-intergalactic-medium

This code was used for the research paper, Deep Forest: Neural Network reconstruction of the Lyman-$\alpha$ forest by Lawrence Huang, Rupert Croft, and Hitesh Arora. The goal of this project is to train a neural network to predict optical depth from noisy flux data. 

## Data
The $\tau$ data is stored in a file named newz3taured.dat. Due to the file's size, we were unable to upload it to Github. [The file is instead available to download through Google Drive. To reach the link, click on this sentence.](https://drive.google.com/file/d/1ozfAwAaDaVaTVMr0NrGLD5aVQdOYqPG4/view?usp=sharing)

## Installation Guide

To install the package written for this project, `lyman_alpha_reconstruction`, you will need to install the following libaries:
<ul>
    <li>torch</li>
    <li>tensorboard
    <li>matplotlib</li>
    <li>numpy</li>
    <li>scipy</li>
</ul>    

## Usage

To be concise, we will import the package as follows:

```python
import lyman_alpha_reconstruction as lar
```

Importing the library is enough to perform data preprocessing. All values and functions will be stored as properties of this module. For example, to access the test dataset for $\tau$ and flux:
```python
tau_test = lar.tauTest
flux_test = lar.fluxTest
```

To create an instance of the neural network:
```python
nn = lar.NetFactory(noise="mid")
```
For this research, we trained and tested neural networks at specific noise levels. A "low" noise level corresponds to an signal-to-noise ratio of 10. A "mid" noise level corresponds to a signal-to-noise ratio of 5. A "high" noise level corresponds to a signal-to-noise ratio of 2.5.

To train the neural network:
```python
nn.run(epochs=50000, learningRate=0.0001)
```
This will train the neural network and write loss values and figures as it trains. These values and figures will be written using tensorboard.

To evaluate the neural network, use the `experimentalRMSEs` function.
```python
lar.experimentalRMSEs(nn, noise="mid")
```
This will write and print RMSE values for $\tau$ predictions by the neural network, a curve-fit version of the neural network output, log (with cubic spline interploation where flux is negative due to noise), and smoothed input log prediction. For the implementation, see `lyman_alpha_reconstruction/reconstruction_methods/log_predictor.py`. Smoothed input log prediction uses the same implementation, except that the input flux values are smoothed first using Gaussian smoothing with a sigma kernel of 6.

These values will be printed and written to the file "statistics.csv"

Note that this file contains the RMSE values used in the paper.