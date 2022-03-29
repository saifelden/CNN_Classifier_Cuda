#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>
#include "size_struct.cpp"
using namespace std;

__global__ void forward(float * input, float* weights,int w_size,Size size,float * output){

    int col = (blockIdx.x*blockDim.x)+threadIdx.x;
    int row = (blockIdx.y*blockDim.y)+threadIdx.y;
    int filter_dim = w_size/2;
    float outval=0;
    int stri =-1;
    for(int i =row-filter_dim;i<=row+filter_dim;i++)
    {
        stri+=1;
        int strj=-1;
        for(int j = col-filter_dim;j<=col+filter_dim;j++)
        {

            strj+=1;
            if(i<0 || j<0)
                continue;
            for(int k=0;k<size.channels;k++)
            {
                outval+=weights[(k*w_size*w_size)+(stri*w_size)+strj]*input[(k*size.width*size.height)+(i*size.height)+j];
            }
        }
    }

    output[(row*size.height)+col]= outval;
}
__global__  void backward(float * input, float* weights,int w_size,Size size,float * deriv_0, float * deriv_1,float lr){
    int col = (blockIdx.x*blockDim.x)+threadIdx.x;
    int row = (blockIdx.y*blockDim.y)+threadIdx.y;
    int filter_dim = w_size/2;
    int stri =-1;
    for(int i =row-filter_dim;i<=row+filter_dim;i++){
        stri+=1;
        int strj=-1;
        for(int j = col-filter_dim;j<=col+filter_dim;j++){
            strj+=1;
            if(i<0 || j<0)
                continue;
            for(int k=0;k<size.channels;k++){
                float curr_w = weights[(k*w_size*w_size)+(stri*w_size)+strj];
                weights[(k*w_size*w_size)+(stri*w_size)+strj] -= (lr*deriv_1[(i*size.height)+j]*input[(k*size.width*size.height)+(i*size.height)+j]);
                deriv_0[(k*size.width*size.height)+(i*size.height)+j]+= deriv_1[(i*size.height)+j]*curr_w;
            }
        }
    }


}
