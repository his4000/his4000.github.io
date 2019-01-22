---
layout: paper
title: "EIE: Efficient Inference Engine on Compressed Deep Neural Network"
date: 2019-01-21
author: Song Han et al.
publish: ISCA 2016
link: https://dl.acm.org/citation.cfm?id=3001163
categories: [research]
---

## Abstract

In the DNN(Deep Neural Network), there are millions of connection between layers. It requires both many computations and memory resources. Especially, the memory access spends so much power to transfer data between DRAM and SRAM.

By reducing the model size, we can reduce the DRAM footprints. That is, we can make an energy efficient accelerator. There is previous research, *Deep Compression (Han et al.)*, which reduces the model size with less accuracy loss. The research reduces the model size in *AlexNet* by 35x, from 240 MB to 6.9 MB.

Total energy savings of EIE are:
- DRAM to SRAM saves 120x
- Exploiting sparsity(Pruning) saves 10x
- Weight sharing gives 8x
- Skipping zero activations from ReLU saves 3x

This paper evaluates 9 DNN benchmarks. EIE is 189x and 13x faster when compared to CPU and GPU implementations of the same DNN without compression.

## Introduction

Large DNN models are very powerful but comsume large amounts of energy because of external memory access. Because the energy consumption depends on external memory access rather than computations. The way to use data reuse or batching is not suitable becuase there is no parameter reuse in *Fully Connected* layer and batching is not suitable in real-time applications.

## Motivation

Matrix-vector multiplication (MxV) is a basic computation block in DNN. But, that is the bottle-neck because there is no reuse of the input matrix. So, you have to access external memory so many times. In CPU and GPU, you can use batching to reduce the number of external memory access. But, in real-time system, that's not suitable.

So, the suitable solution is *Model Compression* to reduce model size. Compression can reduce not only memory accesses but computations. Moreover, this paper suggests to exploit *Dynamic Activation Sparsity* and *Weight Sharing* to Maximize the energy efficiency.

## Main Idea

### Model Compression

In DNN parameters, there are so many zero values after pruning (pruning can reduce model size 10 times smaller by *Deep Compression*). By removing the zeros, we can achieve much smaller size of parameter matrix, *Sparse Matrix*.

![Model Compression](/images/EIE/1.png)
![Memory Layout](/images/EIE/2.png)

In the *Sparse Matrix*, there are 3 kinds of values: Virtual Weight, Relative Row Index, Column Pointer. Virtual weight is real-weight value which is quantized to 4-bits. By **weight sharing** in *Deep Compression*, the parameters can be quantized to 5 or 4 bits without accuracy loss.

Relative row index is the relative location of virtual weight value in vector that each PE has. For example, the green values are in *PE0*. the first relative row index of *PE0* is 0 because there is no zero value before *w0,0*. And the next relative row index is 1 because there is a zero value between *w0,0* and *w8,0*. Let's see 4th relative row index. If the column is changed, relative row index is initialized. So, the 4th relative row index is 1 because there is only one zero before *w4,1* in second column.

Column Pointer means relative location of the cluster group by column. For example, the second value of column index is 3 because there are 3 values which *PE0* has in the first column. And the third value is 4 because there is only 1 value in the second column.