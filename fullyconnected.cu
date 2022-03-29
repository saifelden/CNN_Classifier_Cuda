#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>
#include "size_struct.cpp"

__global__ void calc_sum_block(float * input,float * weights,float* output,int size,int out_index,bool is_forward){
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    __shared__ float* res = new float[blockDim.x];
    __syncthreads();
    if(is_forward)
      res[threadIdx.x] = input[idx]*weights[(out_index*size)+idx];
    else
      res[threadIdx.x] = input[idx]*weights[(idx*size)+out_index];
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

__global__ void FullyConnected::calc_sum_all(float* input,float* result){
    int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    atomicAdd(result,input[idx]);
}
void calc_fc_forward(float * input,int size,int out_size,float* weights,float* output){


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
     calc_sum_block<<<nums_blocks,nums_threads>>>(cinput,cweights,block_sum,size,i,false);

     float* cres,*hres;
     hres = new float[1];
     cudaMalloc(&cres,sizeof(float));
     calc_sum_all<<<1,nums_blocks>>>(block_sum,&cres);
     cudaMemcpy(hres,cres,cudaMemcpyDeviceToHost);
     output[i]=hres[0];
    }

}
 void calc_fc_backward(int size,int out_size,float* weights,float* deriv_0,float* deriv_1){
    int nums_threads = 256;
    int nums_blocks = (out_size/nums_threads)+1;
     float *block_sum,*cweights,*cinput,;
     cudaMalloc(&block_sum,nums_blocks*sizeof(float));
     cudaMalloc(&cweights,size*out_size*sizeof(float));
     cudaMalloc(&cinput,out_size*sizeof(float));
     cudaMemcpy(cweights,weights,cudaMemcpyHostToDevice);
     cudaMemcpy(cinput,deriv_1,cudaMemcpyHostToDevice);
     int nblocks = (nums_blocks/nums_threads)+1;
    for(int i=0;i<size;i++){
     calc_sum_block<<<nums_blocks,nums_threads>>>(cinput,cweights,block_sum,size,i,true);

     float* cres,*hres;
     hres = new float[1];
     cudaMalloc(&cres,sizeof(float));
     calc_sum_all<<<1,nums_blocks>>>(block_sum,&cres);
     cudaMemcpy(hres,cres,cudaMemcpyDeviceToHost);
     deriv_0[i]=hres[0];
    }
}

