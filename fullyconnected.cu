#include "fullyconnected.h"

__global__ void FullyConnected::calc_sum_block(float * input,float * weights,float* output,int size,int out_index){
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    __shared__ float* res = new float[blockDim.x];
    __syncthreads();
    res[threadIdx.x] = input[idx]*weights[(out_index*size)+idx];
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
__global__ float FullyConnected::calc_sum_all(float* input,float* result){
    int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    atomicAdd(result,input[idx]);
}
void calc_fullyconnected::calc_fc_forward(float * input,int size,int out_size,float* weights,float* output){
    int nums_threads = 256;
    int nums_blocks = (size/nums_threads)+1;
     float *block_sum,*cweights,*cinput,;
     cudaMalloc(&block_sum,nums_blocks*sizeof(float));
     cudaMalloc(&cweights,size*out_size*sizeof(float));
     cudaMalloc(&cinput,size*sizeof(float));
     cudaMemcpy(cweights,weights,cudaMemcpyHostToDevice);
     cudaMemcpy(cinput,input,cudaMemcpyHostToDevice);
     int nblocks = (nums_blocks/nums_threads)+1;
    for(int i=0;i<out_size;i++){
     calc_sum_block<<<nums_blocks,nums_threads>>>(cinput,cweights,block_sum,size,i);

     float* cres,*hres;
     hres = new float[1];
     cudaMalloc(&cres,sizeof(float));
     calc_sum_all(block_sum,&cres);
     cudaMemcpy(hres,cres,cudaMemcpyDeviceToHost);
     output[i]=hres[0];
    }
}
