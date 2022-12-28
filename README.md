# I-RAVEN Dataset extended with the Mesh structure
This repository is a fork of a implementation of 

[Stratified Rule-Aware Network for Abstract Visual Reasoning](https://arxiv.org/abs/2002.06838)  
Sheng Hu\*, Yuqing Ma\*, Xianglong Liu†, Yanlu Wei, Shihao Bai  
*Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, 2021  
(\* equal contribution, † corresponding author)

The code for the generation of the I-RAVEN Dataset has been modified with the additional structure called Mesh. The structure is created from the set of lines that follow rules on two attributes - Position and Number.

<div  align="center">    
<img src="https://raw.githubusercontent.com/Adam-Kowalczyk/I-RAVEN-Mesh/master/Images/mesh_example.png" width="70%">
</div> 

## Dataset Generation
Code to generate the dataset resides in the ```I-RAVEN``` folder. The dependencies are consistent with [the original RAVEN](https://github.com/WellyZhang/RAVEN).
* Python 2.7
* OpenCV
* numpy
* tqdm 
* scipy
* pillow

See ```I-RAVEN/requirements.txt``` for a full list of packages required. To install the dependencies, run
```
pip install -r I-RAVEN/requirements.txt
```
To generate a dataset, run
```
python I-RAVEN/main.py --num-samples <number of samples per configuration> --save-dir <directory to save the dataset> --mesh <1 - random, 2 - with rules>
```
Check the ```I-RAVEN/main.py``` file for a full list of arguments you can adjust.




