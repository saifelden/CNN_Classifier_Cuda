#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>
#include "size_struct.cpp"
#include <math.h>
__global__ void calc_softmax_forward(float* input,float total_sum,float* output){
    int idx = (blockDim.x*blockIdx.x)+threadIdx.x;
    output[idx]= exp(input[idx])/total_sum;
}
__global__ void calc_softmax_backward(float* input,float total_sum,float* deriv_0,float * deriv_1){
    int idx = (blockDim.x*blockIdx.x)+threadIdx.x;
    deriv_0[idx]= deriv_1[idx]*exp(input[idx])*(total_sum-exp(input[idx]))/pow(total_sum,2);
}
double calc_crossentropy_forward(float* predicted,int size,float* ground_truth){
    double sum=0.0;
    for(int i=0;i<size;i++)
      sum+=ground_truth[i]*log(predicted[i]);
    sum*=-1;    
    return sum;
}
void calc_crossentropy_backward(float* predicted,int size,float* ground_truth,float* deriv_0){
    for(int i=0;i<size;i++)
       deriv_0[i]=-1*ground_truth[i]/predicted[i];
}

double calc_loss_forward(float * input,int size,float * output,float* ground_truth){
    float exp_sum=0;
    for(int i=0;i<size;i++)
       exp_sum+=exp(input[i]);
    calc_softmax_forward<<<1,size>>>(input,exp_sum,output);
    double loss = calc_crossentropy_forward(output,size,ground_truth);
    return loss;
}
void calc_loss_backward(float* input,float * output,int size,float* deriv_0,float* ground_truth){
    float exp_sum=0;
    for(int i=0;i<size;i++)
       exp_sum+=exp(input[i]);
    float* loss_deriv,*d_input;
    cudaMalloc(loss_deriv,size*sizeof(float));
    cudaMalloc(d_input,size*sizeof(float));
    cudaMemcpy(d_input,input,cudaMemcpyHostToDevice);
    calc_crossentropy_backward(output,size,ground_truth,loss_deriv);
    calc_softmax_backward<<<1,size>>>(d_input,exp_sum,deriv_0,loss_deriv);
}



