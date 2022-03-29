#include<iostream>
#include "convolution.cu"
#include "activation.cu"
using namespace std;
void init_input(float* input,int size,int max_input){
    for(int i=0;i<size;i++){
        if(max_input==1)
         input[i] = (float)(rand()%5783)/54366323.1;
        else  
          input[i]=rand()%max_input;
    }
}
int main(){

    //Testing 2D convolution

    int width = 64;
    int height = 64; 
    int channels = 3;
    Size size;
    size.width=width;
    size.height=height;
    size.channels=channels;
    float * input,* weights,*output,*dinput,*doutput,*dweights,*activation_output;
    input = new float[width*height*channels];
    init_input(input,width*height*channels,256);
    cout<<"input matrix is :"<<endl;
    for(int i=0;i<width*height*channels;i++)
      cout<<input[i]<<" ";
    cout<<endl;
    weights = new float[3*3*channels];
    init_input(weights,3*3*channels,1);
    cout<<"weight matrix is :"<<endl;
    for(int i=0;i<3*3*channels;i++)
      cout<<weights[i]<<" ";
    cout<<endl;
    output = new float[width*height];
    cudaMalloc(&dinput,width*height*channels*sizeof(float));
    cudaMalloc(&activation_output,width*height*channels*sizeof(float));
    cudaMalloc(&doutput,width*height*sizeof(float));
    cudaMalloc(&dweights,3*3*channels*sizeof(float));
    cudaMemcpy(dweights,weights,3*3*channels*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dinput,input,width*height*channels*sizeof(float),cudaMemcpyHostToDevice);

    int thread_size = 16;
    dim3 threads(thread_size,thread_size);
    dim3 blocks(int(64/thread_size),int(64/thread_size));
    forward<<<blocks,threads>>>(dinput,dweights,3,size,doutput);
    calc_tanh_forward<<<blocks,threads>>>(doutput,size,activation_output);
    cudaMemcpy(output,doutput,width*height*sizeof(float),cudaMemcpyDeviceToHost);
    cout<<"output matrix is :"<<endl;
    for(int i=0;i<width*height;i++)
       cout<<output[i]<<" ";
    cout<<endl;
    float * deriv_2 = new float[width*height],*deriv_0 = new float[width*height*channels];
    init_input(deriv_2,width*height,1);
    cout<<"The derivative of the next layer is:  ";
    for(int i=0;i<50;i++)
       cout<<deriv_2[i]<<" ";
    cout<<endl;
    float lr = 0.003;
    float* dderiv_1,*dderiv_0,*dderiv_2;
    cudaMalloc(&dderiv_1,width*height*sizeof(float));
    cudaMalloc(&dderiv_2,width*height*sizeof(float));
    cudaMemcpy(dderiv_2,deriv_2,width*height*sizeof(float),cudaMemcpyHostToDevice);
    cudaMalloc(&dderiv_0,width*height*channels*sizeof(float));
    calc_tanh_backward<<<blocks,threads>>>(doutput,size,dderiv_2,dderiv_1);
    backward<<<blocks,threads>>>(dinput,dweights,3,size,dderiv_0,dderiv_1,lr);
    
    cudaMemcpy(deriv_0,dderiv_0,width*height*channels*sizeof(float),cudaMemcpyDeviceToHost);
    cout<<"The derivative of the previous layer is:  ";
    for(int i=0;i<50;i++)
       cout<<deriv_0[i]<<" ";
    cout<<endl;
    
    return 0;
}