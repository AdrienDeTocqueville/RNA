#include "RNA/Trainers/Supervised.h"

#include "Utility/Error.h"
#include "Utility/Random.h"

#include <limits>
#include "windows.h"

namespace rna
{

void buildBatches(const DataSet& _src, DataSet& _dst, size_t _size)
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

            for (size_t j(0) ; j < _src[0].input.nElements() ; ++j)
                _dst[b].input(i, j) = example.input[j];

            for (size_t j(0) ; j < _src[0].output.nElements() ; ++j)
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

    #ifdef USE_OPENCL
    if (!network->getContext())
        Error::add(ErrorType::USER_ERROR, "OpenCL is necessary for training: call openCL method on network");
    #endif // USE_OPENCL
}

Supervised::~Supervised()
{
    delete loss;
    delete optimizer;
}

#ifdef USE_OPENCL
void Supervised::trainOOO(const DataSet& _dataSet, size_t _steps, size_t _minibatchSize) // NOTE: doesn't work anymore
{
    cl::Context& context = network->getContext();

    cl::CommandQueue inOrder, outOfOrder;
    inOrder.create(context, true);
    outOfOrder.create(context, false);


    DataSet dataSet;
    buildBatches(_dataSet, dataSet, _minibatchSize);

    auto debut = GetTickCount();

    for (size_t step(0); step < _steps; ++step)
    {
        Example& batch = Random::element(dataSet);

        const Tensor& output   = network->feedForward(inOrder, batch.input);
        const Tensor& gradient = loss->getGradient(inOrder, output, batch.output);
        inOrder.join();

        network->backprop(outOfOrder, batch.input, gradient);
        outOfOrder.enqueueBarrier(); // is it necessary ?
        optimizer->updateParams(outOfOrder);

        outOfOrder.join();
    }

    auto time = GetTickCount()-debut;
    std::cout << "Temps: " << (time>1000?time/1000.0f:time) << (time>1000?" s":" ms") << std::endl;

    for (Tensor* param: params)
        outOfOrder.enqueueRead(*param);

    outOfOrder.join();
}

void Supervised::train(const DataSet& _dataSet, size_t _steps, size_t _minibatchSize)
{
    cl::CommandQueue commandQueue;
    commandQueue.create(network->getContext(), true);


    DataSet dataSet;
    buildBatches(_dataSet, dataSet, _minibatchSize);

    auto debut = GetTickCount();

    for (size_t step(0); step < _steps; ++step)
    {
        Example& batch = Random::element(dataSet);

        const Tensor& output = network->feedForward(commandQueue, batch.input);
        const Tensor& gradient = loss->getGradient(commandQueue, output, batch.output);

        network->backprop(commandQueue, batch.input, gradient);
        optimizer->updateParams(commandQueue);

        commandQueue.join();
    }

    auto time = GetTickCount()-debut;
    std::cout << "Temps: " << (time>1000?time/1000.0f:time) << (time>1000?" s":" ms") << std::endl;

    for (Tensor* param: params)
        commandQueue.enqueueRead(*param);

    commandQueue.join();
}

void Supervised::train_generator(Generator _generator, size_t _steps)
{
    auto debut = GetTickCount();

    cl::CommandQueue commandQueue;
    commandQueue.create(network->getContext(), true);


    for (size_t step(0); step < _steps; ++step)
    {
        Example batch = _generator();

        const Tensor& output = network->feedForward(commandQueue, batch.input);
        const Tensor& gradient = loss->getGradient(commandQueue, output, batch.output);

        network->backprop(commandQueue, batch.input, gradient);
        optimizer->updateParams(commandQueue);

        commandQueue.join();
    }

    for (Tensor* param: params)
        commandQueue.enqueueRead(*param);

    commandQueue.join();

    auto time = GetTickCount()-debut;
    std::cout << "Temps: " << (time>1000?time/1000.0f:time) << (time>1000?" s":" ms") << std::endl;
}

void Supervised::earlyStopping(const DataSet& _training, size_t _trainSteps, const DataSet& _testing, size_t _patience, size_t _minibatchSize)
{
    auto debut = GetTickCount();

    cl::CommandQueue commandQueue;
    commandQueue.create(network->getContext(), true);


    DataSet training, testing;
    buildBatches(_training, training, _minibatchSize);
    buildBatches(_testing, testing, _minibatchSize);

    size_t j = 0;
    std::vector<Tensor> bestParams(params.size());
    Tensor::value_type bestError = std::numeric_limits<Tensor::value_type>::max();

    while (j++ < _patience)
    {
        for (size_t step(0); step < _trainSteps; ++step)
        {
            Example& batch = Random::element(training);

            const Tensor& output = network->feedForward(commandQueue, batch.input);
            const Tensor& gradient = loss->getGradient(commandQueue, output, batch.output);

            network->backprop(commandQueue, batch.input, gradient);
            optimizer->updateParams(commandQueue);

            commandQueue.join();
        }


        Tensor::value_type error = validate(testing);

        if (error < bestError)
        {
            std::cout << "Error = " << error << " (new best)" << std::endl;

            bestError = error;
            j = 0;

            for (size_t k(0) ; k < params.size() ; k++)
            {
                commandQueue.enqueueRead(*params[k], CL_TRUE);

                for (size_t k(0) ; k < params.size() ; k++)
                    bestParams[k] = *params[k];
            }
        }
        else
            std::cout << "Error = " << error << " (" << _patience-j << " left)" << std::endl;
    }

    std::cout << std::endl;
    auto time = GetTickCount()-debut;
    std::cout << "Best error: " << bestError << std::endl;
    std::cout << "Time: " << (time>1000?time/1000.0f:time) << (time>1000?" s":" ms") << std::endl;

    // Reload best params
    for (size_t k(0) ; k < params.size() ; k++)
    {
        for (size_t l(0) ; l < bestParams[k].nElements() ; l++)
            (*params[k])[l] = bestParams[k][l];

        commandQueue.enqueueWrite(*params[k]);
    }

    commandQueue.join();
}

void Supervised::earlyStopping_generator(Generator _training, size_t _trainSteps, Generator _testing, size_t _testSteps, size_t _patience)
{
    auto debut = GetTickCount();

    cl::CommandQueue commandQueue;
    commandQueue.create(network->getContext(), true);


    size_t j = 0;
    std::vector<Tensor> bestParams(params.size());
    Tensor::value_type bestError = std::numeric_limits<Tensor::value_type>::max(), errorFactor = 1.0f / _testSteps;

    while (j++ < _patience)
    {
        for (size_t n(0); n < _trainSteps; ++n)
        {
            Example batch = _training();

            const Tensor& output   = network->feedForward(commandQueue, batch.input);
            const Tensor& gradient = loss->getGradient(commandQueue, output, batch.output);

            network->backprop(commandQueue, batch.input, gradient);
            optimizer->updateParams(commandQueue);

            commandQueue.join();
        }


        float error = 0.0f;
        for (size_t n(0); n < _testSteps; ++n)
        {
            Example batch = _testing();

            const Tensor& output = network->feedForward(commandQueue, batch.input);
            commandQueue.enqueueWrite(output, CL_TRUE);

            error += loss->getLoss(output, batch.output);
        }
        error *= errorFactor;

        if (error < bestError)
        {
            std::cout << "Error = " << error << " (new best)" << std::endl;
            bestError = error;
            j = 0;

            for (size_t k(0) ; k < params.size() ; k++)
            {
                commandQueue.enqueueRead(*params[k], CL_TRUE);
                bestParams[k] = *params[k];
            }
        }
        else
            std::cout << "Error = " << error << " (" << _patience-j << " left)" << std::endl;
    }

    std::cout << std::endl;
    auto time = GetTickCount()-debut;
    std::cout << "Best error: " << bestError << std::endl;
    std::cout << "Time: " << (time>1000?time/1000.0f:time) << (time>1000?" s":" ms") << std::endl;

    // Reload best params
    for (size_t k(0) ; k < params.size() ; k++)
    {
        for (size_t l(0) ; l < bestParams[k].nElements() ; l++)
            (*params[k])[l] = bestParams[k][l];

        commandQueue.enqueueWrite(*params[k]);
    }

    commandQueue.join();
}

#else
void Supervised::train(const DataSet& _dataSet, size_t _steps, size_t _minibatchSize)
{
    auto debut = GetTickCount();

    for (size_t step(0); step < _steps; ++step)
    {
        if (step % (_steps / 10) == 0)
            std::cout << "Step: " << step << std::endl;

        for (size_t i(0); i < _minibatchSize; i++)
        {
            const Example& example = Random::element(_dataSet);

            const Tensor& output = network->feedForward(example.input);
            const Tensor& gradient = loss->getGradient(output, example.output);

            network->backprop(example.input, gradient);
        }

        optimizer->updateParams();
    }

    auto time = GetTickCount()-debut;
    std::cout << "Temps: " << (time>1000?time/1000.0f:time) << (time>1000?" s":" ms") << std::endl;
}
#endif // USE_OPENCL

Tensor::value_type Supervised::validate(const rna::DataSet& _testing) const
{
    Tensor::value_type error = 0.0f;

    for (const Example& batch: _testing)
    {
        const Tensor& output = network->feedForward(batch.input);

        error += loss->getLoss(output, batch.output);
    }

    return error / _testing.size();
}

}
