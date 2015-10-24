# Do Deep Neural Networks Learn Facial Action Units When Doing Expression Recognition?
This repository contains all of experiment files for the paper "Do Deep Neural Networks Learn Facial Action Units When Doing Expression Recognition?", available here: http://arxiv.org/abs/1510.02969

![faus_frontpage](./faus_frontpage.png)

## Abstract
Despite being the appearance-based classifier of choice in recent years, relatively few works have examined how much convolutional neural networks (CNNs) can improve performance on accepted expression recognition benchmarks and, more importantly, examine what it is they actually learn. In this work, not only do we show that CNNs can achieve strong performance, but we also introduce an approach to decipher which portions of the face influence the CNN's predictions. First, we train a zero-bias CNN on facial expression data and achieve, to our knowledge, state-of-the-art performance on two expression recognition benchmarks: the extended Cohn-Kanade (CK+) dataset and the Toronto Face Dataset (TFD). We then qualitatively analyze the network by visualizing the spatial patterns that maximally excite different neurons in the convolutional layers and show how they resemble Facial Action Units (FAUs). Finally, we use the FAU labels provided in the CK+ dataset to verify that the FAUs observed in our filter visualizations indeed align with the subject's facial movements. 

### Bibtex
```
@article{khorrami2015deep,
  title={Do Deep Neural Networks Learn Facial Action Units When Doing
         Expression Recognition?},
  author={Khorrami, Pooya and Paine, Tom Le and Huang, Thomas S},
  journal={arXiv preprint arXiv:1510.02969},
  year={2015}
}
```

## About the repo

The experiments are broken up by dataset:

+ ck_plus 
+ ck_plus_six_class
+ tfd

The difference between ``ck_plus`` and ``ck_plus_six_class`` is the ``ck_plus_six_class`` model evaluates our model's performance on the six basic emotions (anger, disgust, fear, happy, sad, surprise) while ``ck_plus`` 
contains the basic six along with neutral and contempt.

The ``README.md`` file in each folder will provide more information on how
to run and evaluate the experiments.

## Requisite Libraries

In order to run our experiments, you will need the following software
