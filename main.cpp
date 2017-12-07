#include <iostream>
#include <ctime>
#include <windows.h>

#include "RNA.h"
#include "Image.h"
#include "Utility/Random.h"

bool envStep(const size_t& action, double& reward, Tensor& nextState, bool v = false);

void loadXOR(unsigned _size, rna::DataSet& _data);
void loadMNIST(unsigned _size, rna::DataSet& _data);

void SARSA();
void SARSALambda();
void QLearning();

Tensor gamestate({1}, 4);

int main()
{
    std::string selection = "GPU";
    std::cout << "Select network (GPU, QL, DQN, XOR, MNIST, CONV, TEST): ";
//    std::cin >> selection;
    std::cout << std::endl;

    for (auto& c: selection)
        c = toupper(c);

    rna::DataSet dataSet;
    rna::Network ann;

    if ("QL" == selection)
    {
        QLearning();
    }

    if ("DQN" == selection)
    {
        unsigned numActions = 2;

        unsigned episodes = 10000;
        unsigned memSize = 10000;
        unsigned batchSize = 64;
        unsigned targetUpdate = 1000;
        double discount = 0.99;

        double epsilonI = 1.0, epsilonF = 0.1;

        unsigned HU = 10;
        ann.addLayer( new rna::Linear(1, HU) );
        ann.addLayer( new rna::Tanh() );
        ann.addLayer( new rna::Linear(HU, 2) );
        ann.addLayer( new rna::Tanh() );

        rna::Network target(ann);

        rna::Memory memory;
        int step = 0;

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
                    memory.push_back({state, action, reward, nextState, terminate});

                else
                    memory[step%memSize] = {state, action, reward, nextState, terminate};

                ann.QLearn(new rna::MSE(), target, memory, batchSize, 0.01, 0.9, discount);
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
    }

    if ("N" == selection)
    {
        loadMNIST(10000, dataSet);

        ann.loadFromFile("Networks/mnist1.rna");

        for (unsigned i(0) ; i < 2000 ; i++)
        {
            i = Random::nextInt(1000, 2000);

            std::cout << std::endl << std::endl << i << std::endl;
            std::cout << ann.feedForward(dataSet[i].input) << std::endl;
            std::cout << ann.feedForward(dataSet[i].input).argmax() << std::endl;
            std::cout << dataSet[i].output << std::endl;

            displayImage(dataSet[i].input, "Output", 8);
        }

//        ann.train(new rna::NLL(), dataSet, 0.001, 0.9, 10000, 500);
//        ann.saveToFile("Networks/mnist1.rna");

//        rna::Reshape r({28*28});
//
//        std::cout << ann.feedForward(r.feedForward(dataSet[1].input)) << std::endl;
//        std::cout << ann.feedForward(r.feedForward(dataSet[1].input)).argmax() << std::endl;
//        std::cout << dataSet[1].output << std::endl;
//
//        std::cout << ann.feedForward(r.feedForward(dataSet[2].input)) << std::endl;
//        std::cout << ann.feedForward(r.feedForward(dataSet[2].input)).argmax() << std::endl;
//        std::cout << dataSet[2].output << std::endl;
//
//        std::cout << ann.feedForward(Matrix({r.feedForward(dataSet[2].input), r.feedForward(dataSet[1].input)})) << std::endl;
    }

    if ("XOR" == selection)
    {
        loadXOR(100, dataSet);

        unsigned HU = 5;
        ann.addLayer( new rna::Linear(2, HU) );
        ann.addLayer( new rna::Tanh() );
        ann.addLayer( new rna::Linear(HU, 1) );
        ann.addLayer( new rna::Tanh() );

        ann.train(new rna::MSE(), dataSet, 0.001, 0.9, 5000, 500);
        ann.saveToFile("Networks/xor.rna");

        std::cout << ann.feedForward(Vector({-0.5, -0.5})) << std::endl; // -1.0
        std::cout << ann.feedForward(Vector({-0.5, 0.5})) << std::endl; // 1.0
        std::cout << ann.feedForward(Vector({0.5, -0.5})) << std::endl; // 1.0
        std::cout << ann.feedForward(Vector({0.5, 0.5})) << std::endl << std::endl; // -1.0

        std::cout << ann.feedForward(Matrix({{-0.5, -0.5}, {-0.5, 0.5}, {0.5, -0.5}, {0.5, 0.5}})) << std::endl << std::endl; // -1.0

        return 0;
    }

    if ("MNIST" == selection)
    {
        loadMNIST(1000, dataSet);
        std::cout << "Loaded" << std::endl;

//        ann.addLayer( new rna::Reshape({28*28}) );
        ann.addLayer( new rna::Linear(28*28, 500) );
        ann.addLayer( new rna::Tanh() );
        ann.addLayer( new rna::Linear(500, 10) );
        ann.addLayer( new rna::Tanh() );
        ann.addLayer( new rna::LogSoftMax() );

        std::cout << "Starting training" << std::endl;

        ann.train(new rna::NLL(), dataSet, 0.01, 0.0, 1000, 100);

        ann.validate(dataSet);
        ann.saveToFile("Networks/mnist1.rna");
    }

    if ("GPU" == selection)
    {
        loadMNIST(1000, dataSet);

        ann.addLayer( new rna::Linear(28*28, 500) );
        ann.addLayer( new rna::Tanh() );
        ann.addLayer( new rna::Linear(500, 10) );
        ann.addLayer( new rna::Tanh() );
        ann.addLayer( new rna::LogSoftMax() );

        ann.toGPU();

        ann.GPUtrain(new rna::NLL(), dataSet, 0.01, 0.0, 1000, 100, 6);
    }

    if ("CONV" == selection)
    {
        loadMNIST(1000, dataSet);
        for (rna::Example& e: dataSet)
            e.input.resize({1, 28, 28});

        std::cout << "DB loaded" << std::endl;

        {
            ann.addLayer( new rna::Convolutional({1, 28, 28}, {5, 5}, 4) );
            ann.addLayer( new rna::MaxPooling() );
            ann.addLayer( new rna::Tanh() );

            ann.addLayer( new rna::Convolutional({4, 12, 12}, {5, 5}, 12) );
            ann.addLayer( new rna::MaxPooling() );
            ann.addLayer( new rna::Tanh() );

            ann.addLayer( new rna::Reshape({12*4*4}) );

            ann.addLayer( new rna::Linear(12*4*4, 500) );
            ann.addLayer( new rna::Tanh() );

            ann.addLayer( new rna::Linear(500, 10) );
            ann.addLayer( new rna::Tanh() );

            ann.addLayer( new rna::LogSoftMax() );
        }

        std::cout << "ANN created" << std::endl;

        ann.train(new rna::NLL(), dataSet, 0.01, 0.3, 10000, 100);

        ann.saveToFile("Networks/leNet1.rna");
    }

    if ("TEST" == selection)
    {
        loadMNIST(2000, dataSet);

        ann.loadFromFile("Networks/leNet1.rna");

        for (unsigned i(0) ; i < 2000 ; i++)
        {
            i = Random::nextInt(1000, 2000);
            dataSet[i].input.resize({1, 28, 28});

            std::cout << std::endl << std::endl << i << std::endl;
            std::cout << ann.feedForward(dataSet[i].input) << std::endl;
            std::cout << ann.feedForward(dataSet[i].input).argmax() << std::endl;
            std::cout << dataSet[i].output << std::endl;

            displayImage(dataSet[i].input, "Output", 8);
        }

        return 0;
    }

    return 0;
}

bool envStep(const size_t& action, double& reward, Tensor& nextState, bool v)
{
    if (action == 0)
        gamestate(0)--;
    if (action == 1)
        gamestate(0)++;

    gamestate(0) = std::min(std::max(Tensor::value_type(0.0), gamestate(0)), Tensor::value_type(8.0));
    nextState = gamestate;


    if (gamestate(0) == 2)
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

    if (gamestate(0) == 2 || gamestate(0) == 6)
    {
    if (v)
        std::cout << "Terminal" << std::endl;
        gamestate(0) = 4;
        return true;
    }

    return false;
}

void loadXOR(unsigned _size, rna::DataSet& _data)
{
    _data.clear();
    _data.resize(_size);

    for (unsigned i(0) ; i < _data.size() ; i++)
    {
        Tensor input{2};
        input.randomize(-1.0, 1.0);

        Tensor output{1};
        if (input(0) * input(1) > 0.0)
            output(0) = -1.0;
        else
            output(0) = 1.0;

        _data[i] = {input, output};
    }
}

void loadMNIST(unsigned _size, rna::DataSet& _data) // 60000 examples
{
    _data.clear();
    _data.resize(_size);

    LoadMNISTImages(_data);
    LoadMNISTLabels(_data);
}

void SARSA()
{
    double Q[9][2] = {0.0};

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

            if (Random::nextDouble() < epsilon)
                action = Random::nextInt(0, 2);

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
        for (unsigned j(0); j < 2; j++)
        {
            std::cout << Q[i][j] << "  ";
        }
        std::cout << std::endl;
    }
}

void SARSALambda()
{
    double lambda = 0.9;

    double Q[9][2] = {0.0};
    double E[9][2] = {0.0};

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
            if (Random::nextDouble() < epsilon)
                action2 = Random::nextInt(0, 2);

            else
                if (Q[(int)nextState(0)][0] > Q[(int)nextState(0)][1])
                    action2 = 0;
            else
                action2 = 1;

            double delta = reward + discount*Q[(int)nextState(0)][action2] - Q[(int)state(0)][action];
            E[(int)state(0)][action]++;

            for (unsigned i(0); i < 9; i++)
            {
                for (unsigned j(0); j < 2; j++)
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
        for (unsigned j(0); j < 2; j++)
        {
            std::cout << Q[i][j] << "  ";
        }
        std::cout << std::endl;
    }
}

void QLearning()
{
    rna::DataSet dataSet;
    rna::Network ann;

    unsigned HU = 10;
    ann.addLayer( new rna::Linear(1, HU) );
    ann.addLayer( new rna::Tanh() );
    ann.addLayer( new rna::Linear(HU, 2) );
    ann.addLayer( new rna::Tanh() );


    Tensor Q({9, 2}, 0.0);

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

            if (Random::nextDouble() < epsilon)
                action = Random::nextInt(0, 2);

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

    for (unsigned i(0); i < 9; i++)
    {
        rna::Example e;
        e.input = Vector({(Tensor::value_type)i});
        e.output = Vector({Q(i, 0), Q(i, 1)});

        dataSet.push_back(e);

        std::cout << dataSet[i].input << ": " << dataSet[i].output << std::endl;
    }

    ann.train(new rna::MSE(), dataSet, 0.001, 0.9, 5000, 5000);

    for (unsigned i(0); i < 9; i++)
    {
        Tensor output = ann.feedForward(Vector({(Tensor::value_type)i})); output.round(2);
        std::cout << Vector({(Tensor::value_type)i}) << ": " << output << std::endl;
    }
}

