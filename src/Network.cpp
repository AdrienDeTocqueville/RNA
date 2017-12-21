#include "RNA.h"

#include "Utility/Random.h"

#include <iostream>
#include <fstream>

#include "windows.h"

//#define RANDOM_SEED

namespace rna
{

void randomMinibatch(const DataSet& _dataSet, Tensor& _inputBatch, Tensor& _outputBatch, const unsigned& _minibatchSize)
{
    _inputBatch.resize({_minibatchSize, _dataSet[0].input.nElements()});
    _outputBatch.resize({_minibatchSize, _dataSet[0].output.nElements()});

    for (size_t i(0); i < _minibatchSize; i++)
    {
        const Example& example = Random::element(_dataSet);

        for (unsigned j(0) ; j < _inputBatch.size(1) ; j++)
            _inputBatch(i, j) = example.input[j];

        for (unsigned j(0) ; j < _outputBatch.size(1) ; j++)
            _outputBatch(i, j) = example.output[j];
    }

    // Restore input structure
    coords_t iSize = _dataSet[0].input.size(); iSize.insert(iSize.begin(), _minibatchSize);
    coords_t oSize = _dataSet[0].output.size(); oSize.insert(oSize.begin(), _minibatchSize);

    _inputBatch.resize(iSize);
    _outputBatch.resize(oSize);
}

Network::Network():
    context(0),
    deviceId(0)
{
    #ifdef RANDOM_SEED
        Random::setSeed();
        std::cout << "Seed: " << Random::getSeed() << std::endl;
    #else
        Random::setSeed(1513874735);
        std::cout << "Warning: seed is fixed to " << Random::getSeed() << std::endl;
    #endif
}

Network::Network(Network&& _network)
{
    swap(*this, _network);
}

Network::Network(const Network& _network):
    context(_network.context),
    deviceId(_network.deviceId)
{
    if (context) // TODO: make this work
        std::cout << "Copy of CL network: This will not work" << std::endl;

    for (const Layer* originalLayer: _network.layers)
    {
        std::string layerType = originalLayer->type;
        Layer* layer = nullptr;


        if ("Linear" == layerType)
            layer = new Linear(*reinterpret_cast<const Linear*>(originalLayer));

        else if ("Reshape" == layerType)
            layer = new Reshape(*reinterpret_cast<const Reshape*>(originalLayer));

        else if ("Convolutional" == layerType)
            layer = new Convolutional(*reinterpret_cast<const Convolutional*>(originalLayer));


        else if ("MaxPooling" == layerType)
            layer = new MaxPooling();

        else if ("Tanh" == layerType)
            layer = new Tanh();

        else if ("ReLU" == layerType)
            layer = new ReLU();

        else if ("LogSoftMax" == layerType)
            layer = new LogSoftMax();

        else
            std::cout << "Unknown layer type: " << layerType << std::endl;


        if (layer)
            layers.push_back(layer);
    }
}

Network::~Network()
{
    releaseCL();

    for (unsigned i(0) ; i < layers.size() ; ++i)
        delete layers[i];
}

Network& Network::operator=(Network _network)
{
    swap(*this, _network);

    return *this;
}

void Network::addLayer(Layer* _layer)
{
    layers.push_back(_layer);
}

Layer* Network::getLayer(size_t _index)
{
    return layers[_index];
}

void Network::openCL(cl_device_type _deviceType)
{
    if (context)
        return;

    cl_int error;

    cl_platform_id platform_id;
    error = clGetPlatformIDs(1, &platform_id, nullptr);
    if (error != CL_SUCCESS)
        std::cout << "Failed to get platforms" << std::endl;

    error = clGetDeviceIDs(platform_id, _deviceType, 1, &deviceId, nullptr);
    if (error != CL_SUCCESS)
        std::cout << "Failed to get devices" << std::endl;

    // Print version
//    cl_uint deviceIdCount = 0;
//    clGetDeviceInfo(deviceId, CL_DEVICE_VERSION, 0, nullptr, &deviceIdCount);

//    char* v = (char*)malloc(deviceIdCount*sizeof(char));
//    clGetDeviceInfo(deviceId, CL_DEVICE_VERSION, deviceIdCount, v, nullptr);
//
//    std::cout << v << std::endl;
//
//    free(v);

    context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &error);
    if (error != CL_SUCCESS)
        std::cout << "Failed to create context" << std::endl;

    for (Layer* l: layers)
        l->openCL(context, deviceId);
}

void Network::releaseCL()
{
    if (!context)
        return;

    clReleaseContext(context);
    context = 0;

    for (Layer* l: layers)
        l->releaseCL();
}

Tensor Network::feedForward(const Tensor& _input)
{
    if (!context)
        return feedForwardCPU(_input);

    else
    {
        std::cout << "can't use openCL with const tensor&" << std::endl;
        return Tensor();
    }
}

Tensor Network::feedForward(Tensor& _input)
{
    if (!context)
        return feedForwardCPU(_input);

    else
        return feedForwardCL(_input);
}

void Network::backprop(const Tensor& _input, const Tensor& _gradOutput)
{
    if (!context)
        return backpropCPU(_input, _gradOutput);

    else
        std::cout << "cna't use openCL with const tensor&" << std::endl;
}

void Network::backprop(Tensor& _input, Tensor& _gradOutput)
{
    if (!context)
        return backpropCPU(_input, _gradOutput);

    else
        return backpropCL(_input, _gradOutput);
}


void Network::zeroParametersGradients()
{
    for (unsigned l(0) ; l < layers.size() ; ++l)
        layers[l]->zeroParametersGradients();
}

void Network::updateParameters(Tensor::value_type _learningRate, Tensor::value_type _inertia)
{
    for (unsigned l(0) ; l < layers.size() ; ++l)
        layers[l]->updateParameters(_learningRate, _inertia);
}

bool Network::saveToFile(const std::string& _file) const
{
    std::ofstream file(_file);

    if (!file)
    {
        std::cout << "Network::saveToFile => Unable to create file: " << _file << std::endl;
        return false;
    }
    else
        std::cout << "Saving network to file: " << _file << std::endl;

    for (const Layer* layer: layers)
    {
        layer->saveToFile(file);

        file << std::endl;
    }

    return true;
}

bool Network::loadFromFile(const std::string& _file)
{
    std::ifstream file(_file);

    if (!file)
    {
        std::cout << "Network::loadFromFile => File not found: " << _file << std::endl;
        return false;
    }
    else
        std::cout << "Loading network from file: " << _file << std::endl;

    if (layers.size())
        std::cout << "Network is not empty: just saying..." << std::endl;

    while (1)
    {
        Layer* layer = nullptr;
        std::string layerType;

        file >> layerType;

        if (file.peek() == EOF)
            break;


        if ("Linear" == layerType)
            layer = new Linear(file);

        else if ("Reshape" == layerType)
            layer = new Reshape(file);

        else if ("Convolutional" == layerType)
            layer = new Convolutional(file);


        else if ("MaxPooling" == layerType)
            layer = new MaxPooling();

        else if ("Tanh" == layerType)
            layer = new Tanh();

        else if ("ReLU" == layerType)
            layer = new ReLU();

        else if ("LogSoftMax" == layerType)
            layer = new LogSoftMax();

        else
            std::cout << "Unknown layer type: " << layerType << std::endl;


        if (layer)
            layers.push_back(layer);
    }

    return true;
}

/// Methods (private)
Tensor Network::feedForwardCPU(const Tensor& _input)
{
    layers.front()->feedForwardCPU(_input);

    for (unsigned l(1) ; l < layers.size() ; ++l)
        layers[l]->feedForwardCPU( layers[l-1]->getOutput() );

    return layers.back()->getOutput();
}

Tensor Network::feedForwardCL(Tensor& _inputBatch)
{
    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, deviceId, nullptr, nullptr);

    _inputBatch.openCL(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

    layers.front()->feedForwardCL(commandQueue, _inputBatch);

    for (unsigned l(1) ; l < layers.size() ; ++l)
        layers[l]->feedForwardCL( commandQueue, layers[l-1]->getOutput() );

    clReleaseCommandQueue(commandQueue);

    return layers.back()->getOutput();
}

void Network::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    layers.back()->backpropCPU(layers[layers.size()-2]->getOutput(), _gradOutput);

    for (unsigned l(layers.size()-2) ; l >= 1 ; l--)
        layers[l]->backpropCPU(layers[l-1]->getOutput(), layers[l+1]->getGradInput());

    layers[0]->backpropCPU(_input, layers[1]->getGradInput());
}

void Network::backpropCL(Tensor& _inputBatch, Tensor& _gradOutputBatch)
{
    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, deviceId, nullptr, nullptr);

    _inputBatch.openCL(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    _gradOutputBatch.openCL(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

    layers.back()->backpropCL(commandQueue, layers[layers.size()-2]->getOutput(), _gradOutputBatch);

    for (unsigned l(layers.size()-2) ; l >= 1 ; l--)
        layers[l]->backpropCL(commandQueue, layers[l-1]->getOutput(), layers[l+1]->getGradInput());

    layers[0]->backpropCL(commandQueue, _inputBatch, layers[1]->getGradInput());

    clReleaseCommandQueue(commandQueue);
}

}
