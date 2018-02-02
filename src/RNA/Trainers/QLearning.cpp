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

    #ifdef USE_OPENCL
    if (!network->getContext())
        Error::add(ErrorType::USER_ERROR, "OpenCL is necessary for training: call openCL method on network");

    commandQueue.create(network->getContext(), true);
    #endif // USE_OPENCL
}

QLearning::~QLearning()
{
    delete loss;
    delete optimizer;
}

#ifdef USE_OPENCL
void QLearning::train(const Memory& _memory, size_t _batchSize)
{
    // Build batch
    std::vector<const Transition*> transitions(_batchSize);
    for (size_t i(0) ; i < _batchSize ; ++i)
        transitions[i] = &Random::element(_memory);

    size_t inputBytes = _memory[0].state.nElements() * sizeof(Tensor::value_type);

    coords_t stateSize = _memory[0].state.size();
    stateSize.insert(stateSize.begin(), _batchSize);

    Tensor stateBatch{stateSize},               nextStateBatch{stateSize};
    stateBatch.openCL(network->getContext());   nextStateBatch.openCL(network->getContext());

    for (size_t i(0); i < _batchSize; ++i)
    {
        commandQueue.enqueueWrite(stateBatch.getBuffer(), CL_TRUE, i*inputBytes, inputBytes, transitions[i]->state.data());
        commandQueue.enqueueWrite(nextStateBatch.getBuffer(), CL_TRUE, i*inputBytes, inputBytes, transitions[i]->nextState.data());
    }

    // Evaluate network
    const Tensor& output = network->feedForward(commandQueue, stateBatch);
    const Tensor& nextOutput = network->feedForward(commandQueue, nextStateBatch);

    commandQueue.enqueueRead(output, CL_FALSE);
    commandQueue.enqueueRead(nextOutput, CL_FALSE);
    commandQueue.join();

    // Compute Q values
    Tensor estimatedQ({_batchSize, 1}, 0.0);
    Tensor targetedQ = nextOutput.max(1);

    for (size_t i(0); i < _batchSize; ++i)
    {
        estimatedQ(i) = output(i, transitions[i]->action);
        targetedQ(i) = transitions[i]->reward + discount * (transitions[i]->terminal? 0.0: targetedQ(i));
    }

//     Compute batch error gradient
    const Tensor& gradient = loss->getGradient(commandQueue, estimatedQ, targetedQ);

    commandQueue.enqueueRead(gradient, CL_FALSE);
    commandQueue.join();

    // Adapt dimensions
    Tensor gradientSparse(output.size(), 0.0);

    for (size_t i(0); i < _batchSize; ++i)
        gradientSparse(i, transitions[i]->action) = gradient(i);

    // Perform backprop
    network->backprop(commandQueue, stateBatch, gradientSparse);
    optimizer->updateParams(commandQueue, _batchSize);

    commandQueue.join();
}

#else
void QLearning::train(const Memory& _memory, size_t _batchSize)
{
    for (size_t i(0) ; i < _batchSize ; ++i)
    {
        const Transition& transition = Random::element(_memory);

        const Tensor& output = network->feedForward(transition.state);
        Tensor expectedValue(output.size(), 0.0), targetedValue(output.size(), 0.0);

        expectedValue(transition.action) = output(transition.action);
        targetedValue(transition.action) = transition.reward + discount * (transition.terminal? 0.0: network->feedForward(transition.nextState).max());

        Tensor gradient = loss->getGradient(expectedValue, targetedValue);

        network->backprop(transition.state, gradient);
    }

    optimizer->updateParams(_batchSize);
}
#endif // USE_OPENCL

}

 /*
unsigned numActions = 2;

unsigned episodes = 10000;
unsigned memSize = 10000;
unsigned batchSize = 64;
unsigned targetUpdate = 1000;

Tensor::value_type epsilonI = 1.0, epsilonF = 0.1;


rna::Network target(ann);

rna::Memory memory;
int step = 0;

for (unsigned i(0); i < episodes; i++)
{
    bool terminate = false;
    Tensor nextState, state = gamestate;
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

        trainer.train(memory, batchSize);
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
