#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>
#include <string>
#include <pair>
#include "size_struct.cpp"
#include<activation.cuh>
#include<batch_norm.cuh>
#include<convolution.cuh>
#include<fullyconnected.cuh>
#include<loss.cuh>
#include<pooling.cuh>
using namespace std;

class ConvolutionNeuralNetwork{

    vector<float**> weights;
    vector<float*> outputs;
    vector<pair<int,int>> w_dims;
    Conv2D conv2d;
    FullyConnected fc;
    BatchNorm bn;
    Pooling pooling;
    Activation activation;
    vector<int> pooling_layers,bn_layers;
    string pooling,activation;

public:
    ConvolutionNeuralNetwork(vector<pair<int,int>> weights_dims,vector<int> pooling_layers,vector<int> bn_layers,string pooling,string activation);
    void calc_forward();
    void calc_backward();


};
