#pragma once

#include "Network.h"

#include "Losses/Loss.h"
#include "Optimizers/Optimizer.h"

#include <iostream>

namespace rna
{

class SGD
{
    public:
        SGD(rna::Network& _network):
            network(&_network),
            loss(nullptr), optimizer(nullptr)
        { }

        virtual ~SGD()
        {
            delete loss;
            delete optimizer;
        }

        void train(const DataSet& _dataSet, unsigned _maxEpochs, unsigned _epochsBetweenReports, unsigned _minibatchSize = 32)
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

        template<typename L, typename... Args>
        void setLoss(Args&&... args)
        {
            loss = new L(args...);
        }

        template<typename O, typename... Args>
        void setOptimizer(Args&&... args)
        {
            optimizer = new O(args...);
        }

    private:
        rna::Network* network;

        Loss* loss;
        Optimizer* optimizer;

        std::vector<Tensor*> params, paramsGrad;
};

}
