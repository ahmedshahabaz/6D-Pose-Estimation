# 6D-Pose-Estimation

## In this project, we reused the code provided by the authors of [Deep Object Pose Estimation for Semantic Robotic Grasping of Household Objects](https://arxiv.org/pdf/1809.10790.pdf).

## The original code can be found in the following [Github](https://github.com/NVlabs/Deep_Object_Pose).

# This project comprised four stages. 

## First, we performed a literature review in which we found that the state-of-the-art approaches are deep learning based. 

## Second, we replicated the results of the DOPE paper with the weights published by the authors. 

## Third, we downloaded the YCB-Video benchmark dataset published in [BOP: Benchmark for 6D Object Pose Estimation](https://bop.felk.cvut.cz/datasets/). Finally, we evaluated the performance of the DOPE model ourselves using the ADD metric proposed in [PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes](https://rse-lab.cs.washington.edu/projects/posecnn/). We had to use the original object models in order to compare the ground truth versus the estimation.
