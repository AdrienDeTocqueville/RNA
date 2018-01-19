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
        QLearning(rna::Network& _network, Tensor::value_type _discount);
        ~QLearning();


        void train();


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
            optimizer = new O(params, paramsGrad, args...);

            optimizer->openCL(network->getContext());
        }

    private:
        rna::Network* network;

        Loss* loss;
        Optimizer* optimizer;

        std::vector<Tensor*> params, paramsGrad;

        Tensor::value_type discount;
};

}
