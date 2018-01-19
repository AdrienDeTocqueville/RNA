#include "RNA.h"

#include <iostream>
#include <fstream>

namespace rna
{

Network::Network()
{ }

Network::Network(Network&& _network)
{
    swap(*this, _network);
}

/*
Network::Network(const Network& _network):
    context(_network.context)
{
    if (context) // FIXME: Network copy
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
*/

Network::~Network()
{
    releaseCL();

    clear();
}

Network& Network::operator=(Network _network)
{
    swap(*this, _network);

    return *this;
}

void Network::addLayer(Layer* _layer)
{
    layers.push_back(_layer);

    if (context)
        _layer->openCL(context);
}

Layer* Network::getLayer(size_t _index)
{
    return layers[_index];
}

void Network::clear()
{
    for (unsigned i(0) ; i < layers.size() ; ++i)
        delete layers[i];

    layers.clear();
}

void Network::openCL(cl_device_type _deviceType)
{
    if (context)
        return;

    context.create(_deviceType);

    for (Layer* l: layers)
        l->openCL(context);
}

void Network::releaseCL()
{
    if (!context)
        return;

    context.release();

    for (Layer* l: layers)
        l->releaseCL();
}

const Tensor& Network::feedForward(const Tensor& _input)
{
    if (!context)
        return feedForwardCPU(_input);

    else
    {
        cl::CommandQueue commandQueue; commandQueue.create(context, true);

        const Tensor& output = feedForwardCL(commandQueue, _input);
        output.readBuffer(commandQueue);

        commandQueue.join();

        return output;
    }
}

void Network::backprop(const Tensor& _input, const Tensor& _outputGrad)
{
    if (!context)
        backpropCPU(_input, _outputGrad);

    else
    {
        cl::CommandQueue commandQueue; commandQueue.create(context, true);
        backpropCL(commandQueue, _input, _outputGrad);
        commandQueue.join();
    }
}

cl::Context& Network::getContext()
{
    return context;
}

void Network::getParams(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad) const
{
    for (Layer* layer: layers)
        layer->getParams(_params, _paramsGrad);
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

        else if ("Convolutional" == layerType)
            layer = new Convolutional(file);


        else if ("LogSoftMax" == layerType)
            layer = new LogSoftMax();

        else if ("MaxPooling" == layerType)
            layer = new MaxPooling(file);

        else if ("Reshape" == layerType)
            layer = new Reshape(file);

        else if ("Dropout" == layerType)
            layer = new Dropout(file);


        else if ("Tanh" == layerType)
            layer = new Tanh();

        else if ("ReLU" == layerType)
            layer = new ReLU();

        else if ("ELU" == layerType)
            layer = new ELU(file);

        else
            std::cout << "Unknown layer type: " << layerType << std::endl;


        if (layer)
            addLayer(layer);
    }

    return true;
}

/// Methods (private)
const Tensor& Network::feedForwardCPU(const Tensor& _input)
{
    layers.front()->feedForwardCPU(_input);

    for (unsigned l(1) ; l < layers.size() ; ++l)
        layers[l]->feedForwardCPU( layers[l-1]->getOutput() );

    return layers.back()->getOutput();
}

const Tensor& Network::feedForwardCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch)
{
    _inputBatch.openCL(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);


    layers.front()->feedForwardCL(_commandQueue, _inputBatch);

    for (unsigned l(1) ; l < layers.size() ; ++l)
        layers[l]->feedForwardCL(_commandQueue, layers[l-1]->getOutput());


    return layers.back()->getOutput();
}

void Network::backpropCPU(const Tensor& _input, const Tensor& _outputGrad)
{
    const Tensor* g = &_outputGrad;

    for (unsigned l(layers.size()-1) ; l >= 1 ; l--)
    {
        layers[l]->backpropCPU(layers[l-1]->getOutput(), *g);
        g = &layers[l]->getInputGrad();
    }

    layers[0]->backpropCPU(_input, *g);
}

void Network::backpropCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch)
{
    _inputBatch.openCL(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    _outputGradBatch.openCL(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);


    const Tensor* g = &_outputGradBatch;

    for (unsigned l(layers.size()-1) ; l >= 1 ; l--)
    {
        layers[l]->backpropCL(_commandQueue, layers[l-1]->getOutput(), *g);
        g = &layers[l]->getInputGrad();
    }

    layers[0]->backpropCL(_commandQueue, _inputBatch, *g);
}

}
