#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>
#include "size_struct.cpp"
using namespace std;
#include<math.h>

__global__ void calc_sigmoid_forward(float *input,Size size,float* output){
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int channels = blockIdx.z*blockDim.z+threadIdx.z;
    float dom = 1.0+(1.0/exp(input[(row*size.width*size.channels)+(col*size.channels)+channels])); 
    output[(row*size.width*size.channels)+(col*size.channels)+channels] = 1/dom;
}

__global__ void calc_sigmoid_backward(float* input,Size size,float* deriv_1,float*deriv_0){
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int channels = blockIdx.z*blockDim.z+threadIdx.z;
    float nom = 1.0/exp(input[(row*size.width*size.channels)+(col*size.channels)+channels]);
    float dom = pow(1.0+nom,2);
    deriv_0[(row*size.width*size.channels)+(col*size.channels)+channels] = deriv_1[(row*size.width*size.channels)+(col*size.channels)+channels]*(nom/dom);
}
__global__ void calc_tanh_forward(float *input,Size size,float* output){
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int channels = blockIdx.z*blockDim.z+threadIdx.z;
    float nom = exp(2.0*input[(row*size.width*size.channels)+(col*size.channels)+channels])-1;
    float dom = exp(2.0*input[(row*size.width*size.channels)+(col*size.channels)+channels])+1;
    output[(row*size.width*size.channels)+(col*size.channels)+channels]=nom/dom;
}
__global__ void calc_tanh_backward(float* input,Size size,float* deriv_1,float*deriv_0)
{
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int channels = blockIdx.z*blockDim.z+threadIdx.z;
    float nom = 4.0*exp(2.0*input[(row*size.width*size.channels)+(col*size.channels)+channels]);
    float dom = pow(exp(2.0*input[(row*size.width*size.channels)+(col*size.channels)+channels])+1.0,2);
    deriv_0[(row*size.width*size.channels)+(col*size.channels)+channels] = deriv_1[(row*size.width*size.channels)+(col*size.channels)+channels]*(nom/dom);
}
