#pragma once

#include <functional>

#include "../Network.h"

#include "../Losses/Loss.h"
#include "../Optimizers/Optimizer.h"

namespace rna
{

struct Example
{
    Tensor input;
    Tensor output;
};

using DataSet = std::vector<Example>;
using Generator = std::function<Example()>;

void buildBatches(const DataSet& _src, DataSet& _dst, size_t _size);


class Supervised
{
    public:
        Supervised(rna::Network& _network);
        ~Supervised();


        void trainOOO(const DataSet& _dataSet, size_t _steps, size_t _minibatchSize = 32);


        void train(const DataSet& _dataSet, size_t _steps, size_t _minibatchSize = 32);
        void train_generator(Generator _generator, size_t _steps);

        void earlyStopping(const DataSet& _training, size_t _trainSteps, const DataSet& _testing, size_t _patience, size_t _minibatchSize = 32);
        void earlyStopping_generator(Generator _training, size_t _trainSteps, Generator _testing, size_t _testSteps, size_t _patience);


        template<typename L, typename... Args>
        void setLoss(Args&&... args)
        {
            delete loss;
            loss = new L(args...);

            #ifdef USE_OPENCL
            loss->openCL(network->getContext());
            #endif // USE_OPENCL
        }

        template<typename O, typename... Args>
        void setOptimizer(Args&&... args)
        {
            delete optimizer;
            optimizer = new O(params, paramsGrad, args...);

            #ifdef USE_OPENCL
            optimizer->openCL(network->getContext());
            #endif // USE_OPENCL
        }

    private:
        Network* network;

        Loss* loss;
        Optimizer* optimizer;

        std::vector<Tensor*> params, paramsGrad;
};

}
