#pragma once

#include "../Network.h"

#include "../Losses/Loss.h"
#include "../Optimizers/Optimizer.h"


namespace rna
{

struct Transition
{
    Tensor state;
    size_t action;
    Tensor::value_type reward;
    Tensor nextState;

    bool terminal;
};

using Memory = std::vector<Transition>;


class QLearning
{
    public:
        QLearning(rna::Network& _network, Tensor::value_type _discount = 0.99);
        ~QLearning();


        #ifdef USE_OPENCL
        void train(const Memory& _memory, size_t _batchSize);
        #else
        void train(const Memory& _memory, size_t _batchSize = 32);
        #endif // USE_OPENCL


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
        rna::Network* network;

        Loss* loss;
        Optimizer* optimizer;

        std::vector<Tensor*> params, paramsGrad;

        Tensor::value_type discount;

        #ifdef USE_OPENCL
        cl::CommandQueue commandQueue;
        #endif // USE_OPENCL
};

}
