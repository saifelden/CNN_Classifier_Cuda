#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>
#include "size_struct.cpp"

__global__ void calc_pooling_forward(float* input,int* chosed,Size size,float *output,char type){
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int channels = blockIdx.z*blockDim.z+threadIdx.z;

    int x = 2*col;
    int y = 2*row;
    int z = channels;
    float tl= input[(y*size.width*size.channels)+(x*size.channels)+z];
    float tr= input[(y*size.width*size.channels)+((x+1)*size.channels)+z];
    float bl= input[((y+1)*size.width*size.channels)+(x*size.channels)+z];
    float br= input[((y+1)*size.width*size.channels)+((x+1)*size.channels)+z];
    
    float maxi;
    if(type=='m')
     maxi= max(tl,max(tr,max(br,bl)));
    else
     maxi= (tl+tr+br+bl)/4;
    if(maxi==tl)
     chosed[(row*size.width*size.channels)+(col*size.channels)+channels]=1;
    else if(maxi==tr)
     chosed[(row*size.width*size.channels)+(col*size.channels)+channels]=2;
    else if(maxi==bl)
     chosed[(row*size.width*size.channels)+(col*size.channels)+channels]=3;
    else
     chosed[(row*size.width*size.channels)+(col*size.channels)+channels]=4;
    
    output[(row*size.width*size.channels)+(col*size.channels)+channels] = maxi;
}
__global__ void calc_maxpooling_backward(float* input,int* chosed,Size size,float* deriv_1,float* deriv_0){
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int channels = blockIdx.z*blockDim.z+threadIdx.z;

    int x = 2*col;
    int y = 2*row;
    int z = channels;
    float tl= input[(y*size.width*size.channels)+(x*size.channels)+z];
    float tr= input[(y*size.width*size.channels)+((x+1)*size.channels)+z];
    float bl= input[((y+1)*size.width*size.channels)+(x*size.channels)+z];
    float br= input[((y+1)*size.width*size.channels)+((x+1)*size.channels)+z];
    deriv_0[(y*size.width*size.channels)+(x*size.channels)+z]=0.0;
    deriv_0[(y*size.width*size.channels)+((x+1)*size.channels)+z]=0.0;
    deriv_0[((y+1)*size.width*size.channels)+(x*size.channels)+z]=0.0;
    deriv_0[((y+1)*size.width*size.channels)+((x+1)*size.channels)+z]=0.0;
    int maxi = chosed[(row*size.width*size.channels)+(col*size.channels)+channels];
    if(max==1)
    deriv_0[(y*size.width*size.channels)+(x*size.channels)+z] = deriv_1[(row*size.width*size.channels)+(col*size.channels)+channels];
   else if(maxi==2)
    deriv_0[(y*size.width*size.channels)+((x+1)*size.channels)+z]=deriv_1[(row*size.width*size.channels)+(col*size.channels)+channels];
   else if(maxi==3)
    deriv_0[((y+1)*size.width*size.channels)+(x*size.channels)+z]=deriv_1[(row*size.width*size.channels)+(col*size.channels)+channels];
   else
    deriv_0[((y+1)*size.width*size.channels)+((x+1)*size.channels)+z]=deriv_1[(row*size.width*size.channels)+(col*size.channels)+channels];
}
__global__ void calc_avgpooling_backward(float* input,Size size,float* deriv_1,float* deriv_0){
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int channels = blockIdx.z*blockDim.z+threadIdx.z;

    int x = 2*col;
    int y = 2*row;
    int z = channels;
    float tl= input[(y*size.width*size.channels)+(x*size.channels)+z];
    float tr= input[(y*size.width*size.channels)+((x+1)*size.channels)+z];
    float bl= input[((y+1)*size.width*size.channels)+(x*size.channels)+z];
    float br= input[((y+1)*size.width*size.channels)+((x+1)*size.channels)+z];
    deriv_0[(y*size.width*size.channels)+(x*size.channels)+z] = 0.25*deriv_1[(row*size.width*size.channels)+(col*size.channels)+channels];
    deriv_0[(y*size.width*size.channels)+((x+1)*size.channels)+z] = 0.25*deriv_1[(row*size.width*size.channels)+(col*size.channels)+channels];
    deriv_0[((y+1)*size.width*size.channels)+(x*size.channels)+z]= 0.25*deriv_1[(row*size.width*size.channels)+(col*size.channels)+channels];
    deriv_0[((y+1)*size.width*size.channels)+((x+1)*size.channels)+z]=0.25* deriv_1[(row*size.width*size.channels)+(col*size.channels)+channels];
}