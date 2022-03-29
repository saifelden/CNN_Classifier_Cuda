#include "batch_norm.cuh"
#include "math.h"
void BatchNorm::BatchNorm(){
}

__global__ void BatchNorm::calc_sum_block(float * input,float* output,Size size){
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    __shared__ float* res = new float[blockDim.x];
    __syncthreads();
    res[idx] = input[idx];
    for(int i = 1 ; i<blockDim.x;i*=2)
    {
        int index = 2*i*threadIdx.x;
        if(index < blockDim.x)
          res[index]+=res[index+i];
        __syncthreads();
    }

    if(threadIdx.x==0)
      output[blockIdx.x] = res[0];
}


__global__ float BatchNorm::calc_sum_all(float* input,float& result){
    int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    float result = 0.0;
    atomicAdd(result,input[idx]);
}

__device__ void BatchNorm::calc_mean(float input,Size size,float & mean){
    mean = input/(size.width*size.height*size.channels);
}
__device__ void BatchNorm::calc_variance(float* input,Size size,float mean,float& variance){
    int col = threadIdx.x +(blockIdx.x*blockDim.x);
    int row = threadIdx.y+ (blockIdx.y*blockDim.y);
    int channels = threadIdx.z+ (blockIdx.z*blockDim.z);
    atomicAdd(variance,(input[(row*size.width*size.channels)+(col*size.channels)+channels]-mean)*(input[(row*size.width*size.channels)+(col*size.channels)+channels]-mean));
}
__device__ void calc_first_phrase(float* input,Size size,float mean, float& first_phrase){
    int col = threadIdx.x +(blockIdx.x*blockDim.x);
    int row = threadIdx.y+ (blockIdx.y*blockDim.y);
    int channels = threadIdx.z+ (blockIdx.z*blockDim.z);
    atomicAdd(first_phrase,(input[(row*size.width*size.channels)+(col*size.channels)+channels]-mean));
}
__global__ void BatchNorm::calc_batchnorm_forward(float * input,float* output,Size size,float& variance,float& mean,float sum_result,float alfa,float beta){
    int col = threadIdx.x + (blockIdx.x*blockDim.x);
    int row = threadIdx.y + (blockIdx.y*blockDim.y);
    int channels  = threadIdx.z + (blockIdx.z*blockDim.z);

    this->calc_mean(sum_result,size,mean);
    this->calc_variance(input,size,mean,variance);
    output[(row*size.width*size.channels)+(col*size.channels)+channels] = (input[(row*size.width*size.channels)+(col*size.channels)+channels]-mean)/sqrt(variance/(size.width*size.height*size.channels));
}

__global__ void BatchNorm::calc_batchnorm_backward(float* input,Size size, float variance,float mean,float* deriv_0,float* deriv_1){
    int col = threadIdx.x + (blockIdx.x*blockDim.x);
    int row = threadIdx.y + (blockIdx.y*blockDim.y);
    int channels  = threadIdx.z + (blockIdx.z*blockDim.z);

    float upper = sqrt(sqrt(variance/(size.width*size.height*size.channels)))*(input[row*size.width*size.channels)+(col*size.channels)+channels]-(1/(size.width*size.height*size.channels)));
    float first_phrase;
    this->calc_first_phrase(input,size,mean,first_phrase);
    float second_phrase = (size.width*size.height*size.channels)*(input[(row*size.width*size.channels)+(col*size.channels)+channels]-mean);
    float backprob = (upper/(second_phrase-first_phrase))*deriv_1[(row*size.width*size.channels)+(col*size.channels)+channels];
    deriv_0[(row*size.width*size.channels)+(col*size.channels)+channels] = backprob;
}
