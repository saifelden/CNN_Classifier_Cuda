#include "cnn.cuh"
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>
using namespace std;
/*
    vector<float**> weights;
    vector<int> pooling_layers,bn_layers;
    string pooling,activation;
*/
ConvolutionNeuralNetwork::ConvolutionNeuralNetwork(vector<pair<int,int>> weights_dims,vector<int> pooling_layers,vector<int> bn_layers,string pooling,string activation,int input_channels){
  weights.resize(weights_dims.size())
  int curr_channels=input_channels;
  for(int i=0;i<weights_dims.size();i++){
      weights[i] = new float[weights_dims[i].first][weights_dims[i].second*weights_dims[i].second*curr_channels];
      curr_channels = weights_dims[i].first;
      for(int j=0;j<weights_dims[i].first;j++){
        curandState *dev_random;
        cudaMalloc((void**)&dev_random, weights_dims[i].second*weights_dims[i].second*curr_channels*sizeof(curandState));
        float *d_weights;
        cudaMalloc(&d_weights,weights_dims[i].second*weights_dims[i].second*curr_channels*sizeof(float));
        int size = weights_dims[i].second*weights_dims[i].second*curr_channels;
        int thread_size = 256;
        initalize_weights<<<(size/thread_size)+1,thread_size>>>(d_weights,dev_random,(i*weights_dims.size())+j);
        cudaMemcpy(weights[i][j],d_weights,cudaMemcpyDeviceToHost);
      }
  }
  this->w_dims = weights_dims
  this->pooling_layers = pooling_layers;
  this->bn_layers = bn_layers;
  this->pooling_str = pooling;
  this->activation_str = activation;
  conv2d = new Conv2D();
  fc = new FullyConnected;
  bn = new BatchNorm();
  pooling= new Pooling();
  activation = new Activation();
}
__global__ ConvolutionNeuralNetwork::initalize_weights(float* kernel,curandState * states,unsigned long iteration){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    curand_init(iteration, idx, 0, &states[idx]);
    kernel[idx] = curand_uniform(&state);
}
void ConvolutionNeuralNetwork::forward(float * input,Size size,){
    float * curr_input = input;
    curr_width = size.width;
    curr_height = size.height;
    curr_channels = size.channels;
    for(int i=0;i<weights.size();i++){
      vector<float *> layer_output;
      
      for(int j=0;j<weights[i].size();j++){
        dim3 threads(16,16);
        float res = (float)curr_width/(float)16;
        if(res%1!=0)
           res=(curr_width/16)+1;
        dim3 blocks(int(res),int(res));
        float * output,*d_output,*d_input,*d_weights;
        cudaMalloc(&d_output,curr_width*curr_height*sizeof(float));
        cudaMalloc(&d_input,curr_width*curr_height*curr_channels*sizeof(float));
        cudaMalloc(&d_weights,w_dims[i].second*w_dims[i].second*curr_channels*sizeof(float));
        cudaMemcpy(d_input,curr_input,cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights,weights[i][j],cudaMemcpyHostToDevice);
        conv2d.forward<<<threads,blocks>>>(d_input,d_weights,w_dims[i].second,size,d_output);
        cudaMemcpy(output,d_output,cudaMemcpyDeviceToHost);
        layer_output.push_back(output);
      }
      int all_size = curr_width*curr_height*weights[j].size()*layer_output.size();
      float * all_output = new float[all_size];
      int pos = 0
      for(int j=0;j<layer_output.size();j++){
          copy(layer_output[j],layer_output[j]+int(curr_width*curr_height*weights[j].size()),all_output+pos);
          pos+=curr_width*curr_height*weights[j].size();
      }
      activation.calc_sigmoid_forward(float *input,Size size,output);
      
      
    }
}