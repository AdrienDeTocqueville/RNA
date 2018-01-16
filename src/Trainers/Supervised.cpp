#include "Supervised.h"

#include <cfloat>

#include "Utility/Error.h"
#include "Utility/Random.h"

#include "windows.h"

namespace rna
{

void buildBatches(const DataSet& _src, DataSet& _dst, unsigned _size)
{
    int nbBatches = _src.size() / _size;
    _dst.resize(nbBatches);

    coords_t iSize = _src[0].input.size(), oSize = _src[0].output.size();
        iSize.insert(iSize.begin(), _size);
        oSize.insert(oSize.begin(), _size);

    for (int b(0); b < nbBatches; ++b)
    {
        _dst[b].input.resize({_size, _src[0].input.nElements()});
        _dst[b].output.resize({_size, _src[0].output.nElements()});

        for (size_t i(0); i < _size; ++i)
        {
            const Example& example = _src[b*_size + i];

            for (unsigned j(0) ; j < _src[0].input.nElements() ; ++j)
                _dst[b].input(i, j) = example.input[j];

            for (unsigned j(0) ; j < _src[0].output.nElements() ; ++j)
                _dst[b].output(i, j) = example.output[j];
        }

        _dst[b].input.resize(iSize);
        _dst[b].output.resize(oSize);
    }
}

Supervised::Supervised(rna::Network& _network):
    network(&_network),
    loss(nullptr), optimizer(nullptr)
{
    network->getParams(params, paramsGrad);

    if (!network->getContext())
        Error::add(ErrorType::USER_ERROR, "OpenCL is necessary for training: call openCL method on network");
}

Supervised::~Supervised()
{
    delete loss;
    delete optimizer;
}

void Supervised::train(const DataSet& _dataSet, unsigned _steps, unsigned _minibatchSize)
{
    auto debut = GetTickCount();

    cl::CommandQueue commandQueue;
    commandQueue.create(network->getContext(), true);


    DataSet dataSet;
    buildBatches(_dataSet, dataSet, _minibatchSize);

    for (unsigned step(0); step < _steps; ++step)
    {
        Example& batch = dataSet[step%dataSet.size()];

        const Tensor& output = network->feedForwardCL(commandQueue, batch.input);
        const Tensor& gradient = loss->getGradientCL(commandQueue, output, batch.output);

        network->backpropCL(commandQueue, batch.input, gradient);
        commandQueue.join();

        optimizer->updateParams(params, paramsGrad);
    }

    for (Tensor* param: params)
        param->readBuffer(commandQueue);

    commandQueue.join();

    auto time = GetTickCount()-debut;
    std::cout << "Temps: " << (time>1000?time/1000:time) << (time>1000?" s":" ms") << std::endl;
}

void Supervised::earlyStopping(const DataSet& _training, const DataSet& _testing, unsigned _steps, unsigned _patience, unsigned _minibatchSize)
{
    auto debut = GetTickCount();

    cl::CommandQueue commandQueue;
    commandQueue.create(network->getContext(), true);


    DataSet training, testing;
    buildBatches(_training, training, _minibatchSize);
    buildBatches(_testing, testing, _minibatchSize);

    unsigned i = 0, j = 0;
    std::vector<Tensor> bestParams(params.size());
    float bestError = FLT_MAX, errorFactor = 1.0f / testing.size();

    while (j++ < _patience)
    {
        for (unsigned step(0); step < _steps; ++step)
        {
            Example& batch = training[(i++)%training.size()];

            const Tensor& output   = network->feedForwardCL(commandQueue, batch.input);
            const Tensor& gradient = loss->getGradientCL(commandQueue, output, batch.output);

            network->backpropCL(commandQueue, batch.input, gradient);
            commandQueue.join();

            optimizer->updateParams(params, paramsGrad);
        }


        float error = 0.0f;
        for (Example& batch: testing)
        {
            const Tensor& output = network->feedForwardCL(commandQueue, batch.input);
            output.readBuffer(commandQueue, CL_TRUE);

            error += loss->getLoss(output, batch.output);
        }
        error *= errorFactor;

        if (error < bestError)
        {
            std::cout << "Error = " << error << " (new best)" << std::endl;
            bestError = error;
            j = 0;

            for (unsigned k(0) ; k < params.size() ; k++)
            {
                params[k]->readBuffer(commandQueue, CL_TRUE);
                bestParams[k] = *params[k];
            }
        }
        else
            std::cout << "Error = " << error << " (" << _patience-j << " left)" << std::endl;
    }

    std::cout << std::endl;
    auto time = GetTickCount()-debut;
    std::cout << "Best error: " << bestError << std::endl;
    std::cout << "Time: " << (time>1000?time/1000:time) << (time>1000?" s":" ms") << std::endl;

    // Reload best params
    for (unsigned k(0) ; k < params.size() ; k++)
    {
        for (unsigned l(0) ; l < bestParams[k].nElements() ; l++)
            (*params[k])[l] = bestParams[k][l];

        params[k]->writeBuffer(commandQueue);
    }

    commandQueue.join();
}

void Supervised::earlyStopping_generator(Generator _training, Generator _testing, size_t _testSize, unsigned _steps, unsigned _patience)
{
    auto debut = GetTickCount();

    cl::CommandQueue commandQueue;
    commandQueue.create(network->getContext(), true);


    unsigned j = 0;
    std::vector<Tensor> bestParams(params.size());
    float bestError = FLT_MAX, errorFactor = 1.0f / _testSize;

    while (j++ < _patience)
    {
        for (unsigned n(0); n < _steps; ++n)
        {
            Example batch = _training();

            const Tensor& output   = network->feedForwardCL(commandQueue, batch.input);
            const Tensor& gradient = loss->getGradientCL(commandQueue, output, batch.output);

            network->backpropCL(commandQueue, batch.input, gradient);
            commandQueue.join();

            optimizer->updateParams(params, paramsGrad);
        }


        float error = 0.0f;
        for (unsigned n(0); n < _testSize; ++n)
        {
            Example batch = _testing();

            const Tensor& output = network->feedForwardCL(commandQueue, batch.input);
            output.readBuffer(commandQueue, CL_TRUE);

            error += loss->getLoss(output, batch.output);
        }
        error *= errorFactor;

        if (error < bestError)
        {
            std::cout << "Error = " << error << " (new best)" << std::endl;
            bestError = error;
            j = 0;

            for (unsigned k(0) ; k < params.size() ; k++)
            {
                params[k]->readBuffer(commandQueue, CL_TRUE);
                bestParams[k] = *params[k];
            }
        }
        else
            std::cout << "Error: " << error << " (" << _patience-j << " left)" << std::endl;
    }

    std::cout << std::endl;
    auto time = GetTickCount()-debut;
    std::cout << "Best error: " << bestError << std::endl;
    std::cout << "Time: " << (time>1000?time/1000:time) << (time>1000?" s":" ms") << std::endl;

    // Reload best params
    for (unsigned k(0) ; k < params.size() ; k++)
    {
        for (unsigned l(0) ; l < bestParams[k].nElements() ; l++)
            (*params[k])[l] = bestParams[k][l];

        params[k]->writeBuffer(commandQueue);
    }

    commandQueue.join();
}

}
