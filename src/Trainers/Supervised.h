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
void buildBatches(const DataSet& _src, DataSet& _dst, unsigned _size);


using Generator = std::function<Example()>;


class Supervised
{
    public:
        Supervised(rna::Network& _network);
        ~Supervised();

        void train(const DataSet& _dataSet, unsigned _steps, unsigned _minibatchSize = 32);

        void earlyStopping(const DataSet& _training, const DataSet& _testing, unsigned _steps, unsigned _patience, unsigned _minibatchSize = 32);
        void earlyStopping_generator(Generator _training, Generator _testing, unsigned _steps, unsigned _patience, size_t _testSize);

        template<typename L, typename... Args>
        void setLoss(Args&&... args)
        {
            delete loss;
            loss = new L(args...);

            loss->openCL(network->getContext());
        }

        template<typename O, typename... Args>
        void setOptimizer(Args&&... args)
        {
            delete optimizer;
            optimizer = new O(args...);

            optimizer->init(params, paramsGrad);
            optimizer->openCL(network->getContext());
        }

    private:
        Network* network;

        Loss* loss;
        Optimizer* optimizer;

        std::vector<Tensor*> params, paramsGrad;
};

}
