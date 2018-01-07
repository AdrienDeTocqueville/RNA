#pragma once

#include "Network.h"

#include "Losses/Loss.h"
#include "Optimizers/Optimizer.h"


namespace rna
{

struct Example
{
    Tensor input;
    Tensor output;
};
using DataSet = std::vector<Example>;

void randomMinibatch(const DataSet& _dataSet, Tensor& _inputBatch, Tensor& _outputBatch, const unsigned& _minibatchSize);


class SGD
{
    public:
        SGD(rna::Network& _network);
        ~SGD();

        void train(const DataSet& _dataSet, unsigned _maxEpochs, unsigned _epochsBetweenReports, unsigned _minibatchSize = 32);

        template<typename L, typename... Args>
        void setLoss(Args&&... args)
        {
            delete loss;
            loss = new L(args...);
        }

        template<typename O, typename... Args>
        void setOptimizer(Args&&... args)
        {
            delete optimizer;
            optimizer = new O(args...);
        }

    private:
        rna::Network* network;

        Loss* loss;
        Optimizer* optimizer;

        std::vector<Tensor*> params, paramsGrad;
};

}
