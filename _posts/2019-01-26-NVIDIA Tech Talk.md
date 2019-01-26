---
layout: post
title: NVIDIA Modern Introduction to CUDA
date: 2019-01-26
categories: [technology]
tags: [NVIDIA, CUDA, Programming, Parallel Programming, Seminar]
---

2019년 1월 25일. 서울 Trade Center에서 CUDA 입문자를 위한 NVIDIA의 입문 세미나가 있었다. 본 세미나는 기존 C/C++ 프로그래밍에 익숙하면서 CUDA를 이용한 GPU 병렬 프로그래밍에 입문하고자 하는 사람을 대상으로 하는 세미나였다.

![NVIDIA Tech Talk](/images/2019-01-26-NVIDIA Tech Talk/1.jpg)

세미나의 서두에서 우리는 CPU와 GPU가 분리된 메모리 사용을 하는 상황을 제외하고 Unified Memory를 사용하는 상황을 가정하여 세미나를 진행하였다.

## Linear Scaling

가장 기본적인 병렬 연산은 무엇인가? 주어진 1차원 배열 [ 1, 2, 3, 4, 5 ]에 2를 곱하여 [ 2, 4, 6, 8, 10 ]으로 만드는 연산을 생각해보자. 이 연산을 병렬 프로세싱을 고려하지 않고 코딩한다고 하면 아래와 같이 코딩할 수 있을 것이다.

~~~c
a[5] = {1, 2, 3, 4, 5};
b[5];

for ( int i = 0 ; i < 5 ; i++ )
  b[i] = a[i] * 2;
~~~

위 코드를 Single Thread로 수행할 경우 배열 a의 각 원소를 순서대로 접근하며 2를 곱하고 배열 b의 각 원소에 순서대로 저장한다. 이 연산을 좀더 자세히 살펴보면 배열 a의 각 원소에 2를 곱하는 연산의 경우 각 연산이 서로 Dependency 없이 수행된다는 것을 알 수 있다. 따라서 우리는 본 연산을 병렬적으로 동시에 수행할 수 있다.

CUDA에서는 단위 연산의 수행을 Thread 개념을 이용해 표현한다. 각 Thread는 병렬 수행의 단위 연산을 수행하며, 이 Thread들을 동시에 수행함으로써 병렬 프로세싱을 할 수 있다. 각 Thread에는 index가 부여되어 각 연산에 접근한다. 병렬 연산의 경우 각 Thread가 연산을 수행하는 시간이 각각 다르므로 코드 내에서 병렬 연산 블록의 마지막에 Synchronization 코드를 넣음으로써 Thread간 동기화를 시켜준다.

1D Linear Scaling을 2D와 3D Linear Scaling까지의 확장은 Thread의 index를 1차원에서 2차원, 3차원으로 확장시킴으로써 가능하다. 하지만 실제 메모리는 1차원으로만 구성되므로 address 연산을 이용하여 1차원 메모리의 data를 2차원으로 매핑할 수 있다.

~~~c
const int N = 5;  // Width of 2D array
int linear_addr;  // 1D memory address
int matrix_addr_x;  // 2D mapping X address
int matrix_addr_y;  // 2D mapping Y address

linear_addr = matrix_addr_y * N + matrix_addr_x;
~~~

CUDA에서는 연산을 Grid 형태로 나누고 한 Grid Cell 내부를 여러개의 Block으로 나누어 Thread를 할당한다. 한 GPU가 가진 프로세서는 정해져 있고, 따라서 병렬적으로 수행시킬 수 있는 연산의 개수도 한정되어 있다. 따라서 CUDA는 한 번에 병렬적으로 수행할 수 있는 연산은 한 Grid Cell에 담아 Block 단위로 수행하고 각 Grid Cell을 순차적으로 수행함으로써 방대한 양의 연산을 수행한다.

## Matrix Transpose

Matrix Transpose는 2차원 배열에서 열과 행을 바꾸는 연산이다. 이 연산은 아주 간단한 예제임과 동시에 GPU의 성능 저하와 최적화를 설명할 수 있는 좋은 예제이다. Matrix Transpose를 코드로 표현하면 아래와 같을 것이다.

~~~c
int a[4][8] = {
                { 1, 2, 3, 4, 5, 6, 7, 8 },
                { 9, 10, 11, 12, 13, 14, 15, 16 },
                { 17, 18, 19, 20, 21, 22, 23, 24 },
                { 25, 26, 27, 28, 29, 30, 31, 32 }
              };
int b[8][4];

for ( int i = 0 ; i < 8 ; i++ ) {
  for ( int j = 0 ; j < 4 ; j++ )
    b[i][j] = a[j][i]
}
~~~

이 코드를 CUDA의 Grid와 Block 개념을 적용하여 표현하면 아래와 같다.

~~~c
const int grid_x = 4;
const int grid_y = 2;
const int block_x = 2;
const int block_y = 2;
int a[4][8] = {
                { 1, 2, 3, 4, 5, 6, 7, 8 },
                { 9, 10, 11, 12, 13, 14, 15, 16 },
                { 17, 18, 19, 20, 21, 22, 23, 24 },
                { 25, 26, 27, 28, 29, 30, 31, 32 }
              };
int b[8][4];

for ( int gy = 0 ; gy < grid_y ; gy++ ) {
  for ( int gx = 0 ; gx < grid_x ; gx++ ) {
    # PARALLELIZE
    for ( int by = 0 ; by < block_y ; by++ ) {
      for ( int bx = 0 ; bx < block_x ; bx++ ) {
        int i = gy * block_y + by;
        int j = gx * block_x + bx;
        b[i][j] = a[j][i];
      }
    }
  }
}
~~~

코드 상에서는 b 배열의 i, j 인덱스를 a 배열과 반대로 넣음으로써 비교적 간단하게 구현됨을 알 수 있다. 그러나 하드웨어적인 관점에서 보면 이는 간단한 연산을 아니다. 실제로 GPU를 이용하여 본 코드를 테스트해보면 GPU가 가진 Max Bandwidth에 한참 미치지 못하는 Bandwidth가 나옴을 알 수 있다. 그 이유는 무엇일까?

Computer Architecture를 공부했던 사람이라면 Spatial Locality라는 개념에 대해 알고 있을 것이다. Spatial Locality란 메모리에서 인접한 위치에 있는 데이터는 함께 사용될 가능성이 높으므로 함께 Caching 되기 때문에 Spatial Locality를 이용하여 코드를 작성하면 Cache hit ratio를 증가시킬 수 있다는 개념이다. 이 관점에서 위 코드를 다시 보면 배열 a는 Spatial Locality를 이용하지 못하고 있다. 배열은 메모리 내에서 행 우선으로 1차원 형태로 저장이 되는데 배열 a의 경우 열 우선으로 접근하는 형태이기 때문이다.

CUDA에서는 이러한 문제를 해결하기 위해 CUDA shared memory를 이용한다. CUDA shared memory란 사용자가 컨트롤 할 수 있는 On-chip memory를 의미한다. CUDA shared memory에 한 Grid Cell의 data를 모두 로드하고, Grid 단위로 CUDA의 Block 연산을 병렬적으로 수행하면 Memory Access를 최적화 시킬 수 있다. 실제로 CUDA shared memory를 이용하여 Matrix Transpose 연산을 다시 수행하면 Bandwidth가 더 개선됨을 볼 수 있다.

Matrix Transpose 연산을 더 최적화 하기 위해서 Memory Bank Parallelization을 수행할 수 있다. 메모리는 data load를 병렬화 시키기 위해 여러개의 Bank를 사용한다. 즉, 여러개의 Bank에서 데이터를 병렬적으로 로드함으로써 데이터 로드 시간을 줄일 수 있다. 배열이 메모리에 저장될 때, Spatial Locality를 극대화하면서 Memory Bank Access를 최적화하기 위해서 인접한 데이터를 서로 다른 Bank에 저장한다. 예를 들어 Bank 개수가 4개인 메모리 시스템의 경우 A[0]은 Bank0에, A[1]은 Bank1에 저장되며, A[4]는 다시 Bank0에 저장되는 방식이다.

한 Grid Cell의 데이터를 로드할 때 같은 Bank에 있는 데이터를 로드하게 되면 Memory Bank Parallelism이 떨어지게 된다. 이를 해결하기 위한 방법은 생각보다 간단한데, 2차원 배열에 한 열을 추가하면 다음 행 데이터가 하나씩 밀리는 현상이 발생하면서 Memory Bank Parallelism이 높아지게 된다.

큰 단위의 연산을 여러개의 작은 연산으로 stride를 나누는 것도 최적화 방법이 될 수 있다. 한 단위의 병렬 연산을 작게 나누어 직렬화 하면 상식적으로 성능이 떨어질 것 같지만 stride를 증가시킴에 따라 성능이 증가하다가 특정 지점에서 부터 감소한다. 그 이유는 GPU의 하드웨어 리소스와 관련이 있다. 예를 들어 GPU의 프로세서가 256개 뿐인데 300개의 연산을 병렬 수행 시키고자 하면 256개의 연산이 먼저 병렬 수행되고 그 이후에 44개의 연산이 병렬 수행된다. 즉, 44개의 연산이 수행되는 동안 212개의 프로세서는 노는 상태가 되는 것이다. 따라서 GPU 리소스에 맞추어 stride를 설정하면 그렇지 않을 때 보다 성능이 증가함을 볼 수 있다. stride가 너무 커지면 오히려 병렬성이 저하되어 다시 성능이 감소한다.

## Conclusion and Wrap-up

평소 CPU를 이용한 직렬화된 프로그래밍만 하는 사람의 경우 병렬 프로그래밍에서 발생하는 이슈를 고려하기 힘들다. 본 Tech Talk에서는 병렬 프로그래밍에서 발생할 수 있는 이슈를 GPU CUDA 프로그래밍을 통해 잘 설명되었다. 병렬 프로그래밍의 이러한 이슈는 비단 GPU CUDA 프로그래밍 뿐만 아니라 병렬 프로세싱을 하는 다른 플랫폼에서도 충분히 참고할 수 있는 내용이다. 특히 요즘 대두되고 있는 AI와 딥러닝 연산은 이러한 병렬 프로세싱의 극대화가 필요한 연산을 요구한다. DNN hardware accelerator를 설계하고, 그 위에서 동작하는 프로그램을 만들 때 본 Tech Talk에서 설명된 이슈들을 이용하면 Performance와 Energy Efficiency를 극대화 시킬 수 있을 것이다.
