# CUDA-Tutorial

## Introduction

Extent https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/, I will add some more examples and explanations to help you understand CUDA programming better.


```bash
nvcc gemmbasic.cu -o a.out
```


```bash
nvcc simpleTensorCoreGEMM.cu -o a.out -lcublas -lcurand -arch=sm_80 # A100
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/yourpath/miniconda3/envs/condaexample/lib ./a.out 
```

## Installation

```bash
conda create -n condaexample python=3.11 #enter later python version if needed
conda activate condaexample 
# Full list at https://anaconda.org/nvidia/cuda-toolkit
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit
```



Reference:
- https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/
- https://github.com/NVIDIA-developer-blog/code-samples
