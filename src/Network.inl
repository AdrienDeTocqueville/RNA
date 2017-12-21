#include <iostream>
#include "windows.h"

#include "Utility/Random.h"

namespace rna
{

template<typename L>
void Network::train(Optimizer<L>& _optimizer, const DataSet& _dataSet, unsigned _maxEpochs, unsigned _epochsBetweenReports, unsigned _minibatchSize)
{
    auto debut = GetTickCount();

    if (!context)
        trainCPU(_optimizer, _dataSet, _maxEpochs, _epochsBetweenReports, _minibatchSize);

    else
        trainCL(_optimizer, _dataSet, _maxEpochs, _epochsBetweenReports, _minibatchSize);

    auto time = GetTickCount()-debut;
    std::cout << "Temps: " << (time>1000?time/1000:time) << (time>1000?" s":" ms") << std::endl;
}

template<typename L>
void Network::trainCPU(Optimizer<L>& _optimizer, const DataSet& _dataSet, unsigned _maxEpochs, unsigned _epochsBetweenReports, unsigned _minibatchSize)
{
    unsigned epoch = 0;
    Tensor::value_type error = 0;

    do
    {
        zeroParametersGradients();

        for (unsigned i(0) ; i < _minibatchSize ; ++i)
        {
            const Example& example = Random::element(_dataSet);

            Tensor output = feedForwardCPU( example.input );

            error += _optimizer.loss.getLoss(output, example.output);
            Tensor gradient = _optimizer.loss.getGradient(output, example.output);

            backpropCPU(example.input, gradient);
        }

        updateParameters(_optimizer.learningRate, _optimizer.inertia);


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
}

template<typename L>
void Network::trainCL(Optimizer<L>& _optimizer, const DataSet& _dataSet, unsigned _maxEpochs, unsigned _epochsBetweenReports, unsigned _minibatchSize)
{
//    _optimizer.loss.openCL(context, deviceId);

    unsigned epoch = 0;
    Tensor::value_type error = 0.0;

    Tensor inputBatch, outputBatch;

    do
    {
        zeroParametersGradients();

        {
            randomMinibatch(_dataSet, inputBatch, outputBatch, _minibatchSize);

            Tensor output = feedForwardCL(inputBatch);

            error += _optimizer.loss.getLoss(output, outputBatch);
            Tensor gradient =_optimizer.loss.getGradient(output, outputBatch);

            // TODO: make openCL usage actually faster for backprop
            backpropCL(inputBatch, gradient);
        }

        updateParameters(_optimizer.learningRate, _optimizer.inertia);


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
}

//template<typename L>
//void Network::QLearn(Optimizer<L>& _optimizer, Network& _target, const Memory& _memory, unsigned _miniBatchSize, double _discount)
//{
//    zeroParametersGradients();
//
//    for (unsigned i(0) ; i < _miniBatchSize ; ++i)
//    {
//        const Transition& transition = Random::element(_memory);
//
//        Tensor output = feedForward(transition.state);
//        Tensor expectedValue(output.size(), 0.0), targetedValue(output.size(), 0.0);
//
//        expectedValue(transition.action) = output(transition.action);
//        targetedValue(transition.action) = transition.reward + _discount * (transition.terminal? 0.0: _target.feedForward(transition.nextState).max());
//
//        Tensor gradient = _optimizer.loss.getGradient(expectedValue, targetedValue);
//
//        backprop(transition.state, gradient);
//    }
//
//    updateParameters(_optimizer.learningRate, _optimizer.inertia);
//}

}
