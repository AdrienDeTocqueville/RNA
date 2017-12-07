#include "RNA.h"

#include "Utility/Random.h"

#include <algorithm>

#include <iostream>
#include <fstream>
#include <ctime>

#include "windows.h"

namespace rna
{

Network::Network():
    context(0),
    deviceId(0)
{
    Random::setSeed(2);
}

Network::Network(Network&& _network)
{
    swap(*this, _network);
}

Network::Network(const Network& _network):
    context(_network.context),
    deviceId(_network.deviceId)
{
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
	clReleaseContext(context);

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

void Network::toGPU()
{
    cl_int error;

    cl_platform_id platform_id;
    error = clGetPlatformIDs(1, &platform_id, nullptr);
	if (error != CL_SUCCESS)
        std::cout << "Failed to get platforms" << std::endl;

    error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &deviceId, nullptr);
	if (error != CL_SUCCESS)
        std::cout << "Failed to get devices" << std::endl;

    context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &error);
	if (error != CL_SUCCESS)
        std::cout << "Failed to create context" << std::endl;

    for (Layer* l: layers)
        l->toGPU(context, deviceId);
}

Tensor Network::feedForward(const Tensor& _input)
{
    if (!layers.size())
        return _input;

    layers.front()->feedForward(_input);

    for (unsigned i(1) ; i < layers.size() ; ++i)
        layers[i]->feedForward( layers[i-1]->output );

    return layers.back()->output;
}

void Network::backprop(const Tensor& _input, const Tensor& _gradOutput)
{
    layers.back()->backprop(layers[layers.size()-2]->output, _gradOutput);

    for (unsigned l(layers.size()-2) ; l >= 1 ; l--)
        layers[l]->backprop(layers[l-1]->output, layers[l+1]->gradInput);

    layers[0]->backprop(_input, layers[1]->gradInput);
}

Tensor Network::GPUfeedForward(const Tensor& _inputBatch)
{
    if (!layers.size())
        return _inputBatch;

    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, deviceId, nullptr, nullptr);


    layers.front()->GPUfeedForward(commandQueue, _inputBatch);

    for (unsigned i(1) ; i < layers.size() ; ++i)
        layers[i]->GPUfeedForward( commandQueue, layers[i-1]->output );


    clReleaseCommandQueue(commandQueue);

    return layers.back()->output;
}

void Network::train(LossFunction* _loss, const DataSet& _dataSet, Tensor::value_type _learningRate, Tensor::value_type _inertia, unsigned _maxEpochs, unsigned _epochsBetweenReports)
{
    auto debut = time(NULL);

    unsigned epoch = 0;
    double error = 0.0;

    _learningRate = std::max(_learningRate, Tensor::value_type(0.0));
    _inertia = std::min(std::max(Tensor::value_type(0.0), _inertia), Tensor::value_type(1.0));

    do
    {
        zeroParametersGradients();

        const Example& example = Random::element(_dataSet);

        Tensor output = feedForward( example.input );

        error += _loss->getLoss(output, example.output);
        Tensor gradient = _loss->getGradient(output, example.output);

        backprop(example.input, gradient);

        updateParameters(_learningRate, _inertia);


        ++epoch;
        if (epoch%_epochsBetweenReports == 0)
        {
            std::cout << "At epoch " << epoch << ":" << std::endl;
            std::cout << "Error = " << error/_epochsBetweenReports << std::endl;

            std::cout << std::endl;

            error = 0.0;
        }
    }
    while (epoch != _maxEpochs);

    delete _loss;

    std::cout << "Temps: " << time(NULL)-debut << std::endl;
}

void Network::GPUtrain(LossFunction* _loss, const DataSet& _dataSet, Tensor::value_type _learningRate, Tensor::value_type _inertia, unsigned _maxEpochs, unsigned _epochsBetweenReports, unsigned _minibatchSize)
{
    auto debut = GetTickCount();

    cl_int error;

    /// Program & kernel
    rna::Linear* l1 = reinterpret_cast<rna::Linear*>(layers[0]);

    /// Buffers
    Tensor inputBatch({_minibatchSize, _dataSet[0].input.nElements()});

    for (size_t i(0); i < inputBatch.size(0); i++)
    {
        const Example& example = Random::element(_dataSet);

        for (unsigned j(0) ; j < inputBatch.size(1) ; j++)
            inputBatch(i, j) = example.input[j];
    }

    std::cout << "Input: " << inputBatch << std::endl;

    debut = GetTickCount();
    std::cout << "Output: " << std::endl << feedForward(inputBatch) << std::endl;
    std::cout << "CPU time: " << GetTickCount()-debut << " ms" << std::endl;

    debut = GetTickCount();
    inputBatch.toGPU(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    std::cout << "Output: " << std::endl << GPUfeedForward(inputBatch) << std::endl;
    std::cout << "GPU time: " << GetTickCount()-debut << " ms" << std::endl;


    delete _loss;


}

void Network::QLearn(LossFunction* _loss, Network& target, const Memory& _memory, Tensor::value_type _learningRate, Tensor::value_type _inertia, unsigned _miniBatchSize, double _discount)
{
    _learningRate = std::max(_learningRate, Tensor::value_type(0.0));
    _inertia = std::min(std::max(Tensor::value_type(0.0), _inertia), Tensor::value_type(1.0));


    zeroParametersGradients();

    for (unsigned i(0) ; i < _miniBatchSize ; ++i)
    {
//        const Transition& transition = Random::element(_memory);
//
//        Tensor output = feedForward(transition.state);
//        Tensor expectedValue(output.size(), 0.0), targetedValue(output.size(), 0.0);
//
//        expectedValue(transition.action) = output(transition.action);
//        targetedValue(transition.action) = transition.reward + _discount * (transition.terminal? 0.0: target.feedForward(transition.nextState).max());
//
//        Tensor gradient = _loss->getGradient(expectedValue, targetedValue);
//
//        backprop(transition.state, gradient);
    }

    updateParameters(_learningRate, _inertia);
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

void Network::validate(const DataSet& _dataSet)
{
    double correct = 0.0;

    for (auto& example: _dataSet)
    {
        Tensor output = feedForward(example.input);

        if (output.argmax()[0] == example.output(0))
            correct++;
    }

    std::cout << "Validation: " << correct << " over " << _dataSet.size() << " examples" << std::endl;
}

bool Network::saveToFile(std::string _file) const
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

        file << std::endl << std::endl;
    }

    return true;
}

bool Network::loadFromFile(std::string _file)
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

}
