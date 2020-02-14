# StarGAN
A Tensorflow implementation of StarGAN

https://arxiv.org/abs/1711.09020

# Requirements
* tensorflow 1.15
* matplotlib
* pandas

# Dataset
[celebA dataset from kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset)

# Usage
 ## Train
 `python main.py --mode train`
 ## Test
 `python main.py --mode test --test_img_path path_to_test_img --selected_attributes Male Young --attr_values 0 1`

# Usage

# References
Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, Jaegul Choo, <em>StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation</em>, 2017
