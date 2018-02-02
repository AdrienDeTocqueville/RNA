#include "RL.h"
#include "Utility/Random.h"

#include <iostream>

namespace RL
{

Tensor gamestate({1}, 4);
const int numActions = 2;

void test()
{
    std::cout << std::endl << "=== Testing RL ===" << std::endl;


    rna::Network ann;

    #ifdef USE_OPENCL
    ann.openCL(cl::DeviceType::CPU);
    std::string name = "netCL.rna";
    #else
    std::string name = "net.rna";
    #endif


    std::string selection = "TRAIN";

    if ("RL" == selection)
        QLearning();

    else if ("LOAD" == selection)
    {
        ann.loadFromFile("res/Networks/"+name);

        Tensor input = Tensor{7, 1};
        for (unsigned i(0); i < input.size(0); i++)
            input(i, 0) = i;

        Tensor output = ann.feedForward(input); output.round(4);
        std::cout << input << std::endl << output << std::endl;
    }

    else
    {
        unsigned HU = 6;
        ann.add( new rna::Linear(1, HU) );
        ann.add( new rna::Tanh() );
        ann.add( new rna::Linear(HU, numActions) );
        ann.add( new rna::Tanh() );

        rna::QLearning trainer(ann);
            trainer.setLoss<rna::Huber>();
            trainer.setOptimizer<rna::RMSProp>(0.001f);



        unsigned episodes = 10000;
        unsigned memSize = 1000;
        Tensor::value_type epsilonI = 1.0, epsilonF = 0.1;

        rna::Memory memory; memory.reserve(memSize);
        int step = 0;

        for (unsigned i(0); i < episodes; i++)
        {
            bool terminate = false;
            Tensor nextState, state = gamestate;
            size_t action;
            double reward;

            while (!terminate)
            {
                double epsilon = std::max(0.0, epsilonI - step*(epsilonI-epsilonF)/(2.0*episodes));

                if (Random::next<double>() < epsilon)
                    action = Random::next(0, numActions);
                else
                {
                    #ifdef USE_OPENCL
                    Tensor batch = state;
                    batch.resize({1, state.nElements()});
                    action = ann.feedForward(batch).argmax()[0];
                    #else
                    action = ann.feedForward(state).argmax()[0];
                    #endif
                }

                terminate = envStep(action, reward, nextState);

                if (memory.size() < memSize)
                    memory.push_back({state, action, (Tensor::value_type)reward, nextState, terminate});

                else
                    memory[step%memSize] = {state, action, (Tensor::value_type)reward, nextState, terminate};

                trainer.train(memory, 32);
//                system("pause");

                state = nextState;

                step++;
//                if (step % targetUpdate == 0)
//                    target = rna::Network(ann);
            }
        }

        ann.saveToFile("res/Networks/"+name);


        #ifdef USE_OPENCL

        Tensor input = Tensor{7, 1};
        for (unsigned i(0); i < input.size(0); i++)
            input(i, 0) = i;

        Tensor output = ann.feedForward(input); output.round(4);
        std::cout << input << std::endl << output << std::endl;
        #else
        for (unsigned i(0); i < 7; i++)
        {
            Tensor input = Vector({(Tensor::value_type)i});
            Tensor output = ann.feedForward(input); output.round(4);
            std::cout << input << ": " << output << std::endl;
        }
        #endif // USE_OPENCL
    }
}
bool envStep(const size_t& action, double& reward, Tensor& nextState, bool v)
{
    if (action == 0)
        gamestate(0)--;
    if (action == 1)
        gamestate(0)++;

    gamestate(0) = std::min(std::max(Tensor::value_type(0), gamestate(0)), Tensor::value_type(6));
    nextState = gamestate;


    if (gamestate(0) == 0)
    {
        reward = -1;
    }
    else if (gamestate(0) == 6)
    {
        reward = 1;
    }
    else
        reward = 0;

    if (v)
    std::cout << "State: " << gamestate(0) << std::endl;
    if (v)
    std::cout << "Reward: " << reward << std::endl;

    if (gamestate(0) == 0 || gamestate(0) == 6)
    {
        if (v)
            std::cout << "Terminal" << std::endl;

        gamestate(0) = Random::next(1, 6);
        return true;
    }

    return false;
}

void SARSA()
{
    double Q[9][numActions] = {0.0};

    unsigned episodes = 10000;
    double discount = 0.99;

    double epsilonI = 1.0, epsilonF = 0.5;
    int step = 0;

    for (unsigned episode(0); episode < episodes; episode++)
    {
        bool terminate = false;
        Tensor state = gamestate, nextState;
        size_t action = 1;
        double reward;

        while (!terminate)
        {
            double epsilon = std::max(0.0, epsilonI - step*(epsilonI-epsilonF)*0.0001);
            double alpha = 0.01;

            if (Random::next<double>() < epsilon)
                action = Random::next<int>(0, numActions);

            else
                if (Q[(int)state(0)][0] > Q[(int)state(0)][1])
                    action = 0;
            else
                action = 1;

            terminate = envStep(action, reward, nextState);

            size_t action2 = 1;
            if (Q[(int)nextState(0)][0] > Q[(int)nextState(0)][1])
                    action2 = 0;

            Q[(int)state(0)][action] += alpha * (reward + discount*Q[(int)nextState(0)][action2] - Q[(int)state(0)][action]);
            state = nextState;

            step++;
        }
    }

    for (unsigned i(0); i < 9; i++)
    {
        for (unsigned j(0); j < numActions; j++)
        {
            std::cout << Q[i][j] << "  ";
        }
        std::cout << std::endl;
    }
}

void SARSALambda()
{
    double lambda = 0.9;

    double Q[9][numActions] = {0.0};
    double E[9][numActions] = {0.0};

    unsigned episodes = 10000;
    double discount = 0.99;

    double epsilonI = 1.0, epsilonF = 0.5;
    int step = 0;

    for (unsigned episode(0); episode < episodes; episode++)
    {
        bool terminate = false;
        Tensor state = gamestate, nextState;
        size_t action = 0;
        double reward;

        while (!terminate)
        {
            terminate = envStep(action, reward, nextState);

            double epsilon = std::max(0.0, epsilonI - step*(epsilonI-epsilonF)/(double)episodes);
            double alpha = 0.001;

            size_t action2;
            if (Random::next<double>() < epsilon)
                action2 = Random::next<int>(0, numActions);

            else
                if (Q[(int)nextState(0)][0] > Q[(int)nextState(0)][1])
                    action2 = 0;
            else
                action2 = 1;

            double delta = reward + discount*Q[(int)nextState(0)][action2] - Q[(int)state(0)][action];
            E[(int)state(0)][action]++;

            for (unsigned i(0); i < 9; i++)
            {
                for (unsigned j(0); j < numActions; j++)
                {
                    Q[(int)state(0)][action] += alpha * delta * E[(int)state(0)][action];
                    E[(int)state(0)][action] *= discount * lambda;
                }
            }

            state = nextState;
            action = action2;

            step++;
        }
    }

    for (unsigned i(0); i < 9; i++)
    {
        for (unsigned j(0); j < numActions; j++)
        {
            std::cout << Q[i][j] << "  ";
        }
        std::cout << std::endl;
    }
}

void QLearning()
{
    Tensor Q({9, numActions}, 0.0);

    unsigned episodes = 10000;
    double discount = 0.99;

    double epsilonI = 1.0, epsilonF = 0.1;
    int step = 0;

    for (unsigned episode(0); episode < episodes; episode++)
    {
        bool terminate = false;
        Tensor state = gamestate, nextState;
        size_t action;
        double reward;

        while (!terminate)
        {
            double epsilon = std::max(0.0, epsilonI - step*(epsilonI-epsilonF)/(double)episodes);
            double alpha = 0.01;

            if (Random::next<double>() < epsilon)
                action = Random::next<int>(0, numActions);

            else
                if (Q(state(0), 0) > Q(state(0), 1))
                    action = 0;
            else
                action = 1;

            terminate = envStep(action, reward, nextState);

            size_t action2 = 1;
            if (Q(nextState(0), 0) > Q(nextState(0), 1))
                    action2 = 0;

            Q(state(0), action) += alpha * (reward + discount*Q(nextState(0), action2) - Q(state(0), action));
            state = nextState;

            step++;
        }
    }

    std::cout << Q << std::endl;
}

}