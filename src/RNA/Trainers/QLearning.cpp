#include "RNA/Trainers/QLearning.h"

#include "Utility/Error.h"
#include "Utility/Random.h"

#include <cfloat>
#include "windows.h"

namespace rna
{

QLearning::QLearning(rna::Network& _network, Tensor::value_type _discount):
    network(&_network),
    loss(nullptr), optimizer(nullptr),
    discount(_discount)
{
    network->getParams(params, paramsGrad);

    if (!network->getContext())
        Error::add(ErrorType::USER_ERROR, "OpenCL is necessary for training: call openCL method on network");
}

QLearning::~QLearning()
{
    delete loss;
    delete optimizer;
}

void QLearning::train()
{
}

}

/*
unsigned numActions = 2;

unsigned episodes = 10000;
unsigned memSize = 10000;
unsigned batchSize = 64;
unsigned targetUpdate = 1000;
Tensor::value_type discount = 0.99;

Tensor::value_type epsilonI = 1.0, epsilonF = 0.1;

unsigned HU = 10;
ann.add( new rna::Linear(1, HU) );
ann.add( new rna::Tanh() );
ann.add( new rna::Linear(HU, 2) );
ann.add( new rna::Tanh() );

rna::Network target(ann);

rna::Memory memory;
int step = 0;

rna::Optimizer<rna::MSE> op(0.01f, 0.0f);

for (unsigned i(0); i < episodes; i++)
{
    bool terminate = false;
    Tensor state = gamestate, nextState;
    size_t action;
    double reward;

    while (!terminate)
    {
        double epsilon = std::max(0.0, epsilonI - step*(epsilonI-epsilonF)/(double)episodes);

        if (Random::nextDouble() < epsilon)
            action = Random::nextInt(0, numActions);
        else
            action = ann.feedForward(state).argmax()[0];

        terminate = envStep(action, reward, nextState);

        if (memory.size() < memSize)
            memory.push_back({state, action, (Tensor::value_type)reward, nextState, terminate});

        else
            memory[step%memSize] = {state, action, (Tensor::value_type)reward, nextState, terminate};

//                ann.QLearn(op, target, memory, batchSize, discount);
        state = nextState;

        step++;
        if (step % targetUpdate == 0)
            target = rna::Network(ann);
    }
}

//        for (unsigned i(0); i < 9; i++)
//        {
//            Tensor input = Vector({(double)i});
//            Tensor output = ann.feedForward(input); output.round(2);
//            std::cout << input << ": " << output << std::endl;
//        }
*/
