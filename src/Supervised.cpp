#include "Supervised.h"

#include "Utility/Error.h"
#include "Utility/Random.h"

#include "windows.h"

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
}

Supervised::~Supervised()
{
    delete loss;
    delete optimizer;
}

void Supervised::train(const DataSet& _dataSet, unsigned _maxEpochs, unsigned _epochsBetweenReports, unsigned _minibatchSize)
{
    auto debut = GetTickCount();

    cl::Context& context = network->getContext();
    if (!context)
    {
        Error::add(ErrorType::USER_ERROR, "OpenCL is necessary for training");
        return;
    }

    unsigned epoch = 0;
    Tensor::value_type error = 0.0;
    cl::CommandQueue commandQueue;

    commandQueue.create(context, true);

    optimizer->init(params, paramsGrad);

    loss->openCL(context);
    optimizer->openCL(context);

    do
    {
        Tensor inputBatch, outputBatch;
        randomMinibatch(_dataSet, inputBatch, outputBatch, _minibatchSize);

//        // Temporary solution
//        // Problem: CL buffer would not updated on device
//        inputBatch.releaseCL();
//        outputBatch.releaseCL();

        const Tensor& output = network->feedForwardCL(commandQueue, inputBatch);
        const Tensor& gradient = loss->getGradientCL(commandQueue, output, outputBatch);

        network->backpropCL(commandQueue, inputBatch, gradient);
        commandQueue.join();

        optimizer->updateParams(commandQueue, params, paramsGrad);
        commandQueue.join();

        output.readBuffer(commandQueue, CL_TRUE);
        error += loss->getLoss(output, outputBatch);

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

    for (Tensor* param: params)
        param->readBuffer(commandQueue);

    commandQueue.join();

    auto time = GetTickCount()-debut;
    std::cout << "Temps: " << (time>1000?time/1000:time) << (time>1000?" s":" ms") << std::endl;
}

void Supervised::earlyStopping(const DataSet& _training, const DataSet& _testing, std::function<unsigned(Network&, DataSet&)> _validate, unsigned _steps, unsigned _patience, unsigned _minibatchSize)
{
    auto debut = GetTickCount();

    cl::Context& context = network->getContext();
    if (!context)
    {
        Error::add(ErrorType::USER_ERROR, "OpenCL is necessary for training");
        return;
    }

    cl::CommandQueue commandQueue;
    commandQueue.create(context, true);

    optimizer->init(params, paramsGrad);

    loss->openCL(context);
    optimizer->openCL(context);

    DataSet training, testing;
    buildBatches(_training, training, _minibatchSize);
    buildBatches(_testing, testing, _minibatchSize);

    unsigned i = 0, j = 0;
    unsigned bestError = UINT_MAX;
    std::vector<Tensor> bestParams(params.size());

    while (j++ < _patience)
    {
        for (int n(0); n < _steps; ++n)
        {
            Example& batch = training[(i+n)%training.size()];

            const Tensor& output   = network->feedForwardCL(commandQueue, batch.input);
            const Tensor& gradient = loss->getGradientCL(commandQueue, output, batch.output);

            network->backpropCL(commandQueue, batch.input, gradient);
            commandQueue.join();

            optimizer->updateParams(commandQueue, params, paramsGrad);
            commandQueue.join();
        }

        i += _steps;

//        unsigned error = 0;
//        for (unsigned k(0) ; k < testing.size() ; ++k)
//        {
//            Example& batch = testing[k];
//
//            const Tensor& output = network->feedForwardCL(commandQueue, batch.input);
//            output.readBuffer(commandQueue, CL_TRUE);
//
//            for (unsigned b(0) ; b < _minibatchSize ; b++)
//            {
//                for (unsigned l(0) ; l < output.size(1) ; l++)
//                {
//                    if (l != batch.output(b, 0) && output(b, l) >= output(b, batch.output(b, 0)))
//                    {
//                        error++;
//                        break;
//                    }
//                }
//            }
//        }

        unsigned error = _validate(*network, testing);
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

    auto time = GetTickCount()-debut;
    std::cout << "Temps: " << (time>1000?time/1000:time) << (time>1000?" s":" ms") << std::endl;
    std::cout << "Best error: " << bestError << std::endl;

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
