#pragma once

#include "Network.h"

#include "Losses/Loss.h"
#include "Optimizers/Optimizer.h"

#include <functional>

namespace rna
{

struct Example
{
    Tensor input;
    Tensor output;
};
using DataSet = std::vector<Example>;

void randomMinibatch(const DataSet& _dataSet, Tensor& _inputBatch, Tensor& _outputBatch, const unsigned& _minibatchSize);
void buildBatches(const DataSet& _src, DataSet& _dst, unsigned _size);


class Supervised
{
    public:
        Supervised(rna::Network& _network);
        ~Supervised();

        void train(const DataSet& _dataSet, unsigned _maxEpochs, unsigned _epochsBetweenReports, unsigned _minibatchSize = 32);
        void earlyStopping(const DataSet& _training, const DataSet& _testing, std::function<unsigned(Network&, DataSet&)> _validate, unsigned _steps, unsigned _patience, unsigned _minibatchSize = 32);

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
        Network* network;

        Loss* loss;
        Optimizer* optimizer;

        std::vector<Tensor*> params, paramsGrad;
};

}
