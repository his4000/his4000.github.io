---
layout: post
title: 2019 GDG Deep Learning Summer School
date: 2019-09-02
categories: [technology]
tags: [DNN, Seminar]
published: true
---

2019년 8월 31일, 광주과학기술원 (GIST) 에서 열린 GDG (Google Developers Group) Deep Learning Summer School에 참석하였다. 본 세미나는 딥러닝 기술을 대표적으로 한 인공지능 기술 소개와 GDG 구성원들의 경험을 공유하는 자리였다. 본 포스팅의 내용은 당일 진행된 강연 내용을 기반으로 작성되었다.

# DL; Deep Learning

**Speaker: Yong Yi Lee, GDG Organizer**

## Machine Learning

Machine Learning은 데이터를 기반으로 한 인공지능 기술의 한 갈래이다. 방대한 데이터로부터 일반화 된 규칙을 만들어내고, 이를 문제 해결에 사용하는 기술이다. 데이터 셋에 대한 통계적 분석이 기반이 되며, 때문에 고품질의 방대한 데이터가 고성능의 Machine Learning 시스템을 만든다. Machine Learning은 Supervised Learning과 Unsupervised Learning으로 구분된다.

**Supervised Learning**은 주어진 데이터 샘플에 대응되는 추론 결과(Label)가 명확히 주어진 문제를 풀기 위한 Machine Learning 기술이다. 결과값이 연속적 분포를 가지는 경우 Regression 방식을 이용하여 모델링하고, 이산적 분포를 가지는 경우 Classification 방식을 이용하여 모델링한다.

**Unsupervised Learning**은 주어진 데이터 샘플에 대응되는 추론 결과가 명확하지 않은 문제를 풀기 위한 기술로, 데이터 패턴을 수학적 원리에 의거하여 학습하는 경우가 많다.

**Reinforce Learning**은 데이터가 없거나 부족한 상황에서 *Reward* 시스템을 이용하여 학습 모델링을 진행하는 방법이다. 주어진 Environment에서 Actor가 행위를 반복하고 그에 대한 *Reward*를 제공하여 학습을 진행한다.

### Linear Regression

Machine Learning 수행 방법 중 **Linear Regression**은 주어진 데이터로부터 선형 모델 가설을 세우고, 이를 점진적으로 개선하는 과정을 거쳐 모델링을 진행하는 것을 말한다. Linear Regression은 총 세 단계의 과정으로 이루어진다.

1. Hypothesis : Linear Model 가설 설정
2. Cost Function : 비용 함수 설정
3. Optimization : Gradient Descent 알고리즘을 이용한 최적화

먼저 해당 데이터셋으로부터 기대되는 가설(Linear Model)을 세운다. 이후에 데이터셋을 이용하여 프로세싱을 수행하고, 그 결과를 수집한다. 가설에 따른 결과와 실제 결과 사이의 비용(Cost)를 Cost Function을 세워 산출한다. 마지막으로 Cost가 최소화 되는 방향으로 모델을 최적화하면서 모델의 정확도를 높인다.

### Gradient Descent

**Gradient Descent** 알고리즘은 모델의 최적화 과정에서 사용되는 알고리즘이다. 미분에 기반한 수학적 접근 방법으로, Cost의 최소값이 되는 모델링 지점을 미분을 이용해 근사하는 방법으로 진행된다. 이 때, 초기값, 즉, 초기 모델링을 어떻게 지정하느냐가 중요한데, 초기 모델링이 잘못될 경우 학습 시간이 오래 걸리거나 최적의 모델링 결과를 내지 못하는 문제가 발생할 수 있다.

또한, Gradient Descent 과정에서 Local Minima에 빠지는 경우가 발생할 수 있는데 이는 Convex Cost Function을 이용하여 완화할 수 있다.

Gradient Descent을 진행하는 과정에서 Learning Rate를 어떻게 설정하느냐도 중요한 이슈이다. Learning Rate가 과도하게 클 경우 최적 모델링 지점으로 수렴하지 않고 발산하는 경우가 발생하며, 과도하게 작을 경우 학습 시간이 오래 걸리는 단점이 발생한다.

### Model Evaluation

모델링을 마친 후 모델에 대한 검증을 거치는 것도 중요하다. 모델링 검증 단계에서 주요 주안점은 나의 모델이 다른 많은 경우도 잘 예측을 하는가이다. 쉽게 말해 Training에 사용한 데이터셋을 이용하여 검증을 하게 되면, 해당 데이터는 Training 과정에서 잘 학습된 데이터이므로, 당연히 높은 예측 정확도를 낼 것이다. 그러나 실제 Machine Learning의 활용에서는 Training 데이터셋 이외의 데이터에서도 높은 예측 정확도를 내야한다.

Evaluation 단계에서의 신뢰도를 위해 우리는 우리가 가진 데이터셋을 Training set과 Test set으로 나눌 필요가 있다. Training 단계에서는 Training set을 이용하여 학습을 진행하고 Evaluation 단계에서 Test set을 이용하여 Training 상황에서 주어진 경우 이외의 상황에 대하여 검증해야 한다.

## Artificial Neural Network

**Artificial Neural Network** 란 인간의 신경망 구조를 모사하여, Non-Linear한 가설 모델링을 위한 Machine Learning의 한 종류이다. 인간의 신경망 구조에서 뉴런들이 순차적으로 신호를 전달하는 구조와 신호 자극의 세기가 일정 수준 이상일 경우 신호를 전달한다는 것을 모방하여 설계되었다.

Artificial Neural Network는 *Synapse*, *Neuron cell body*, *Activation function*으로 이루어진 *Perceptron*들이 다양한 계층을 이루어 연결된 형태이다. 

Linear Regression에서처럼 Artificial Neural Network에서도 Gradient Descent가 적용되는데, 이를 위해 **Back Propagation**이라는 방법이 고안되었다. Back Propagation은 Gradient Descent를 Forward Propagation 의 역방향으로 순차적으로 적용하면서 각 Perceptron을 최적의 모델로 근사하는 방법이다. 

이 과정에서 **Vanishing Gradient**라고 하는 문제가 발생하는데, 이는 *sigmoid* 함수를 Activation Function으로 사용함으로써 sigmoid 함수에 대해 미분이 지속적으로 진행되면서 propagation power가 점차 감소하는 현상이다. 이를 해결하기 위해 **ReLU** 함수가 대세로 사용되기 시작하였다.

## Deep Learning

**Deep Learning**은 여러 개의 Perceptron 들이 깊은 네트워크를 이룬 Artificial Neural Network를 말한다. 현대에 들어서 Deep Learning이 발전할 수 있었던 원동력으로 세 가지 요인이 지목되는데,

1. Big data
2. GPU computing
3. Improved algorithm

위의 세 가지 요인이 있다. Deep Learning은 Machine Learning의 한 종류로써 Deep Learning 또한 데이터에 기반한 모델이다. 따라서 방대한 데이터와 이에 대한 분석 기술의 발달이 Deep Learning 발전에 큰 영향을 미쳤다고 할 것이다.

GPU는 Deep Learning의 기본이 되는 Perceptron 연산을 병렬적으로 수행하기에 최적화된 구조를 가지고 있다. 이런 GPU computing의 발전은 Deep Learning을 실제로 수행하는데 많은 도움을 주었다.

또한 다양한 Deep Learning 알고리즘의 발전은 Deep Learning에 기반한 문제 해결을 더 빠르게 만들어 주었다.

### Improved Algorithms

#### Weight Initialization

초기 모델링에 대한 중요성에 대해 앞서 언급하였다. **Weight Initialization**도 같은 맥락에서 중요한 과정이다. 초기 weight 값이 global optima에 멀리 초기화 될 경우 학습 시간이 오래 걸리고, overfitting 등의 문제가 발생할 수 있다.

#### Batch Normalization

**Batch Normalization** 또한 현재 많이 사용되는 방법으로, 각 perceptron의 입력 데이터를 normalize하여 학습 효율을 높이는 방법이다.

이 외에도 **Drop out**, **Ensemble**, **Adam Optimization**등에 대한 설명이 있었다.

여기 까지의 Deep Learning에 대한 개략적인 설명이 약 1시간 30분 정도 이루어졌다. 그 이후 세션에서는 Deep Learning 특정 분야에 대한 세미나가 2개의 트랙으로 이루어졌다. 이에 대해서 간략하게 정리해 보겠다.

# Track II. RL; Reinforcement Learning

**Speaker : Ju Sung Kang, GIST**

**Reinforcemenr Learning** (강화 학습)이란 데이터가 주어지지 않거나 부족한 상황에서 반복적인 경험을 통해 학습을 진행하는 방법이다.

Reinforcement Learning에서는 주어진 *Environment*와 Environment 안에서 Action을 취하는 *Actor*가 있고, Actor의 Action에 대해 양수와 음수의 *Reward*를 제공함으로써 학습이 진행된다.

## Episode

**Episode**란 Actor가 Environment 안에서 수행한 일련의 Action과 그 결과 총 Reward를 총체적으로 말한다. Reinforcement Learning에서는 이 Episode가 하나의 학습 단위가 된다. 즉, Actor가 여러 가지 episode를 수행하고, 그 결과 reward가 가장 높은 episode를 선택하는 방향으로 학습을 진행한다.

## MDP (Markov Decision Processing)

강화 학습을 수행하기 위한 첫 단계로 **MDP**를 정의해야 한다. MDP란 해당 Environment에서의 *state*, *action*, *reward*로 이루어진 그래프로, 전체 episode의 방향을 보여준다.

## Deep Q Learning

MDP를 만든 후, 각 episode에 대하여 total reward가 최대가 되는 방향을 결정하기 위해 **Q-function**을 이용한 **Deep Q Learning**을 수행한다. Q-function이란 *Greedy Algorithm*을 이용하여 reward가 최대가 되는 방향의 action을 선택하는 함수로, *Q-table*을 업데이트하는 방식으로 최적화를 진행한다.

Q-table을 업데이트하는 과정에서 두 가지 이슈가 있는데,

### Exploitation

업데이트 된 reward 결과를 최대한 반영하여 학습을 진행한다.

### Exploration

업데이트 된 reward 외의 action 방향에 대해서도 학습을 진행 해본다. 이는 학습의 방향이 초기의 reward에 의해 지나치게 편중되는 것을 방지한다.

이외의 이슈로 **Data Correlation** 이슈도 들었다.

### Data Correlation

**Data Correlation**이란 강화 학습에서 연속된 학습 데이터가 유사할 경우 학습의 방향이 편중되는 것을 의미한다. 특히 이러한 유사성이 초기에 반복될 경우, 모델은 초기 유사 패턴을 절대적으로 따라가는 잘못을 저지를 수 있다. 이를 방지하기 위해 데이터셋을 충분히 섞어 학습을 진행해야 한다.

# Track I. RNN; Recurrent Neural Network 

**Speaker : Rak Hoon Son, GIST**

**RNN**은 주로 시간 순서대로 나타나는 데이터셋에 대한 딥러닝 모델이다. RNN은 같은 구조의 perceptron이 반복되는 형태를 가지고 있으며, CNN이 공간적 데이터를 기반으로 하는 것과 비교된다. 

## LSTM

RNN에서도 Vanishing Gradient 문제가 발생하는데, 그 원인은 시계열 데이터에서 오랜 과거의 데이터를 반영하는데 어려운 구조를 가지기 때문이다. **LSTM**은 이를 보완하기 위한 구조로, 오랜 과거의 데이터도 Local Memory에 저장하여 현재 연산에 반영하는 구조를 가지고 있다.

이를 통해 Vanishing Gradient 문제를 부분적으로 해결할 수 있으나, 요구되는 Memory 양이 크고, 연산이 복잡해지는 문제를 가지고 있다.

또한 과거의 데이터에 대하여 동일한 가중치로 학습을 시키는 문제도 가지고 있다. Vanishing Gradient의 문제가 발생하기는 하지만 과거의 데이터는 현재에 더 적은 영향을 미칠 가능성이 높다. 따라서 **Forget Gate**를 이용하여 과거 데이터의 영향을 조절한다.

## ATTM

LSTM이 과거의 결과를 저장하기는 하지만 모든 데이터의 저장이 어렵고, 따라서 비교적 인접한 데이터를 저장한다는 단점은 여전히 Vanishing Gradient를 발생시킨다. **ATTM** 은 더욱 깊은 RNN 모델에서 먼 과거의 데이터를 직접적으로 현재에 반영시킴으로써, Vanishing Gradient를 더욱 급진적으로 개선한다.

이 과정에서 과거로 부터의 data path에 scoring을 함으로써 그 영향력을 조정한다. Scoring 방법에는 *Dot-product*나 *Scaled-dot-product*를 사용한다.



이 외에도 **TRAN**이나 **BERT**에 대한 설명도 이어졌다.

# Track I. GAN; Generative Adversarial Network

**Speaker : Sung Han Lee, GIST**

**GAN**은 요즘 많은 주목을 받고 있는 기술로, 쉽게 말해 데이터를 만들어내는 딥러닝 모델이다. GAN을 이용하여 data labeling을 하기도 하고, 이를 통해 *Unsupervised Learning*에 기여하기도 한다.

## Nash Equilibrium

GAN은 기본적으로 *Generator*와 *Discriminator*의 경쟁으로 동작한다. Generator는 모조의 데이터를 만들어내는 방법을 학습하고, Discriminator는 Generator가 만들어낸 모조의 데이터를 식별하는 것을 학습한다. 둘의 경쟁을 통해 최종적으로 **Nash Equilibrium**에 도달하는 것이 GAN의 목표이다.

## Problem

GAN을 수행하는 과정에는 수많은 문제점들이 발생한다.

- **Partial Collapse** : Generator가 생성하는 데이터 중 누락된 데이터가 발생함을 의미한다.
- **Unwanted Sample** : 찌그러진 사진 등 원하지 않는 데이터를 만들어낸다.
- **Balancing** : Generator가 낮은 비용의 이미지만을 생성한다. 이는 Discriminator의 overfitting을 초래한다.

## Solution

GAN의 여러 문제점을 해결하기 위해 발전된 형태의 GAN이 제시되었다.

- **Unrolled GAN** : Generator에게 Discriminator의 진행 방향을 알려줌으로써 *Partial Collapse* 문제를 해결한다.
- **WGAN** : 비용 함수를 변경하여 *Balancing* 문제를 해결한다.
- **Progressive GAN** : 작은 이미지 사이즈에서 시작하여 점점 이미지 사이즈를 키우는 방향으로 학습을 진행한다. 높은 해상도의 결과물을 생성할 수 있으며, 빠르게 결과물을 생성할 수 있다.

## Measurement

GAN으로 학습한 모델을 어떻게 평가할 것인가도 중요한 이슈이다.

#### Amazon Mechanical Turk

이 방법은 GAN 모델의 학습 결과를 여러 사람에게 평가를 의뢰하는 방법으로, 집단 지성을 이용한 대표적인 방법이다. 이 방법은 기계가 해결하지 못하는 부분을 사람이 직접 해결할 수 있다는 장점이 있으나, 평가 의뢰에 대한 비용이 많이 발생하고, 평가 결과의 정확성을 보장하기 힘들다는 단점이 있다. 또한 학습 모델의 도메인이 매우 전문적인 분야일 경우 평가자를 모집하는데 어려움이 있을 수 있다.

#### Inception Score

이 방법은 GoogLeNet에서 사용된 **Inception Layer**를 이용하여 GAN 모델을 평가하는 방식이다. Inception Layer를 통과한 이미지는 원래의 이미지와 비슷한 형태를 보인다는 것에 기반한 방법으로, 기계를 이용한 평가가 가능하기 때문에 Amazon Mechanical Turk에서 발생하는 단점을 보완할 수 있다.

#### Frechet Inception Distance (FID)

현재 가장 대세로 자리잡고 있는 방법이다. *Precision*과 *Recall*이라는 두 가지 항목을 이용하는데, Precision은 유사도를 측정하고, Recall은 다양성을 측정하는 방식으로 진행된다.

#### Neuroscore

뇌 과학과의 접목을 통해 해결하는 방법이다. GAN이 생성한 이미지를 보고 나서 발생하는 뇌파를 측정하여 뇌파의 변화 부분을 이용하여 평가하는 방식이다.

## Application

GAN의 활용 방향에 대해서는 강연자가 전공한 뇌 과학과 관련한 방향이 주를 이루었다. 뇌 과학 지식의 부족으로 인해 자세한 내용의 이해는 어려웠으나, 뇌 과학과 결합하여 인간의 생각을 읽어낼 수 있다는 주장이 흥미로웠다. 과연 인공지능의 발전은 인간의 뇌 속을 들여다보는 수준까지 갈 것인가?



**GAN**에 대한 강연은 상당히 심화된 내용으로 구성되어 이해하고 정리하는데 다소 어려움이 있었다. 그러나 GAN에 대한 주제와 연구 방향이 흥미로운 부분이 많았고, 추후 GAN에 대해 공부한 이후에 본 세미나의 내용을 더 잘 이해할 수 있을 것 같다.

# Track II : GNN; Graph Neural Network

**Speaker : Dong Hyun Kim**

**GNN**은 그래프 자료구조 형태를 가진 데이터에 대하여 분석하는 모델이다. 그래프 형태의 데이터라고 함은 대표적으로 Social Network Community 데이터나 Brain Signal Graph 등이 있다.

GNN은 그래프 자료구조를 Edge와 Vertex 각각의 행렬로 표현함으로써, 딥러닝 연산을 수행할 수 있도록 한다. GNN의 프로세싱 과정은 아래 세 단계로 이루어진다.

1. **Aggregate** : 이웃한 vertex로 부터 정보를 수집한다.
2. **Combine** : Aggregate에서 수집한 정보와 대상 vertex의 정보를 종합한다.
3. **Readout** : Graph의 시작점에 따라 모델링 결과가 달라질 수 있으므로, 이에 대한 영향력을 줄여준다.

## Graph Convolutional Network

**GCN (Graph Convolutional Network)**는 이웃한 weight를 사용하는 *CNN (Convolutional Neural Network)*와 GNN의 특징을 연관지어 지역적인 데이터 재사용을 극대화하는 방법이다.

이 방식에는 **Graph Laplacian**이나 **ChebNet**과 같은 수학적인 접근이 주로 사용된다.

본 세미나에서 설명한 GNN 방식은 몇 가지 한계점을 가지고 있는데,

1. Computational cost가 높다.
2. 고정된 그래프에만 적용된다. Social Network Graph의 경우 시간의 경과에 따라 변형된다.
3. Undirected 그래프에만 적용된다.

# Wrap-up

어렴풋이 들어왔던 딥러닝 각 분야에 대해서 총체적으로 이해하기에 좋은 시간이었다. 전체적으로 딥러닝 입문자도 이해할 수 있는 내용으로 구성하기 위해 노력한 듯 보였다. 또한, 연구 공간에서 벗어나 다른 연구자들은 어떤 주제에 대해서 연구하고 생각하고 있는지 알기에 좋은 시간이었다. GDG (Google Developers Group)에 대해서도 처음 접했는데 그들이 가지고 있는 비전과 의미가 상당히 공감되었고, 향후 활동에 대해서도 많은 기대가 되었다. 다만, 수많은 내용을 하루에 다 다루기 위해 내용의 깊이를 포기한 부분이 많은 것 같아 그 부분은 아쉬움으로 남았다.