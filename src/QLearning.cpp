#include "SGD.h"

#include "Utility/Random.h"

#include <iostream>

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

SGD::SGD(rna::Network& _network):
    network(&_network),
    loss(nullptr), optimizer(nullptr)
{
    network->getParams(params, paramsGrad);
}

SGD::~SGD()
{
    delete loss;
    delete optimizer;
}

void SGD::train(const DataSet& _dataSet, unsigned _maxEpochs, unsigned _epochsBetweenReports, unsigned _minibatchSize)
{
    cl::Context& context = network->getContext();

    unsigned epoch = 0;
    Tensor::value_type error = 0.0;
    cl::CommandQueue commandQueue;

    Tensor inputBatch, outputBatch;

    commandQueue.create(context, true);

    optimizer->init(params, paramsGrad);

    loss->openCL(context);
    optimizer->openCL(context);

    do
    {
        randomMinibatch(_dataSet, inputBatch, outputBatch, _minibatchSize);

        // Temporary solution
        // Problem: CL buffer is not updated
        inputBatch.releaseCL();
        outputBatch.releaseCL();

        const Tensor& output = network->feedForwardCL(commandQueue, inputBatch);
        const Tensor& gradient = loss->getGradientCL(commandQueue, output, outputBatch);

        network->backpropCL(commandQueue, inputBatch, gradient);

        commandQueue.join();

        optimizer->updateParams(commandQueue, params, paramsGrad);

        commandQueue.join();

        ++epoch;
        if (epoch%_epochsBetweenReports == 0)
        {
            std::cout << "At epoch " << epoch << ":" << std::endl;
            std::cout << "Error = " << error/(_epochsBetweenReports*_minibatchSize) << std::endl;

            std::cout << std::endl;

            error = 0.0;
        }
    }
    while (epoch != _maxEpochs);

    for (Tensor*& param: params)
        param->readBuffer(commandQueue);

    commandQueue.join();
}

}
