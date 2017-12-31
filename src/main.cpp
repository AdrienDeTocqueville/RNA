#include <iostream>
#include <ctime>
#include <windows.h>

#include "RNA.h"
#include "Image.h"
#include "Utility/Random.h"

bool envStep(const size_t& action, double& reward, Tensor& nextState, bool v = false);

void loadXOR(unsigned _size, rna::DataSet& _data);
void loadMNIST(unsigned _size, rna::DataSet& _data);

void validateMNIST(rna::Network& _ann, rna::DataSet& _dataSet);

void SARSA();
void SARSALambda();
void QLearning();

Tensor gamestate({1}, 4);
const auto deviceType = CL_DEVICE_TYPE_ALL;

int main()
{
    std::string selection = "N";

    if (selection.empty())
    {
        std::cout << "Select training set (MNIST, CONV, XOR, RL, DQN, TEST): ";
        std::cin >> selection;
    }
    else
        std::cout << "Selected training set: " << selection;
    std::cout << std::endl;

    for (auto& c: selection)
        c = toupper(c);

    rna::DataSet dataSet;
    rna::Network ann;

    if ("RL" == selection)
        QLearning();

    if ("DQN" == selection)
    {
        unsigned numActions = 2;

        unsigned episodes = 10000;
        unsigned memSize = 10000;
        unsigned batchSize = 64;
        unsigned targetUpdate = 1000;
        Tensor::value_type discount = 0.99;

        Tensor::value_type epsilonI = 1.0, epsilonF = 0.1;

        unsigned HU = 10;
        ann.addLayer( new rna::Linear(1, HU) );
        ann.addLayer( new rna::Tanh() );
        ann.addLayer( new rna::Linear(HU, 2) );
        ann.addLayer( new rna::Tanh() );

//        rna::Network target(ann);

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
//                if (step % targetUpdate == 0)
//                    target = rna::Network(ann);
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
        std::string device = "GPU";

        loadMNIST(100, dataSet);

        ann.addLayer( new rna::Reshape({28*28}, device == "GPU") );
        ann.addLayer( new rna::Linear(28*28, 500) );
        ann.addLayer( new rna::Tanh() );
        ann.addLayer( new rna::Linear(500, 10) );
        ann.addLayer( new rna::Tanh() );
        ann.addLayer( new rna::LogSoftMax() );

        if ("GPU" == device)
            ann.openCL(deviceType);

        Tensor inputB, outputB;
        rna::randomMinibatch(dataSet, inputB, outputB, 1);

//        Tensor inputB = dataSet[0].input;

        std::cout << inputB << std::endl;
        std::cout << ann.feedForward(inputB) << std::endl;

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

        rna::Optimizer<rna::MSE> op(0.001f, 0.9f);

        ann.train(op, dataSet, 5000, 500);
        ann.saveToFile("res/Networks/xor.rna");

        std::cout << ann.feedForward(Vector({-0.5, -0.5})) << std::endl; // -1.0
        std::cout << ann.feedForward(Vector({-0.5, 0.5})) << std::endl; // 1.0
        std::cout << ann.feedForward(Vector({0.5, -0.5})) << std::endl; // 1.0
        std::cout << ann.feedForward(Vector({0.5, 0.5})) << std::endl << std::endl; // -1.0

        std::cout << ann.feedForward(Matrix({{-0.5, -0.5}, {-0.5, 0.5}, {0.5, -0.5}, {0.5, 0.5}})) << std::endl << std::endl; // -1.0

        return 0;
    }

    if ("MNIST" == selection)
    {
        std::string device = "CPU";
//        std::cout << "Use openCL (CPU/GPU) ? "; std::cin >> device; std::cout << std::endl;

        loadMNIST(10000, dataSet);

        ann.addLayer( new rna::Reshape({28*28}, device == "GPU") );
        ann.addLayer( new rna::Linear(28*28, 500) );
        ann.addLayer( new rna::Tanh() );
        ann.addLayer( new rna::Linear(500, 10) );
        ann.addLayer( new rna::Tanh() );
        ann.addLayer( new rna::LogSoftMax() );

        if ("GPU" == device)
            ann.openCL(deviceType);

        std::cout << "Starting training on " << device << std::endl;

        rna::Optimizer<rna::NLL> op(0.01f, 0.1f);

        ann.train(op, dataSet, 1000, 100, 32);
        ann.saveToFile("res/Networks/mnist" + device + ".rna");

        validateMNIST(ann, dataSet);
    }

    if ("CONV" == selection)
    {
        loadMNIST(10000, dataSet);
        for (rna::Example& e: dataSet)
            e.input.resize({1, 28, 28});

        {
            ann.addLayer( new rna::Convolutional({1, 28, 28}, {5, 5}, 4) );
            ann.addLayer( new rna::MaxPooling() );
            ann.addLayer( new rna::Tanh() );

            ann.addLayer( new rna::Convolutional({4, 12, 12}, {5, 5}, 12) );
            ann.addLayer( new rna::MaxPooling() );
            ann.addLayer( new rna::Tanh() );

            ann.addLayer( new rna::Reshape({12*4*4}, true) );

            ann.addLayer( new rna::Linear(12*4*4, 500) );
            ann.addLayer( new rna::Tanh() );

            ann.addLayer( new rna::Linear(500, 10) );
            ann.addLayer( new rna::Tanh() );

            ann.addLayer( new rna::LogSoftMax() );
        }

        std::cout << "ANN created" << std::endl;

        rna::Optimizer<rna::NLL> op(0.01f, 0.1f);

        ann.train(op, dataSet, 1000, 100, 32);

        ann.saveToFile("res/Networks/leNet1.rna");
    }

    if ("TEST" == selection)
    {
        loadMNIST(100, dataSet);
        for (rna::Example& e: dataSet)
            e.input.resize({1, 28, 28});

        ann.addLayer( new rna::Reshape({1, 1, 28, 28}) );
        ann.addLayer( new rna::Convolutional({1, 28, 28}, {5, 5}, 12) );
        ann.addLayer( new rna::MaxPooling() );
        ann.addLayer( new rna::Tanh() );
        ann.addLayer( new rna::Reshape({1, 12, 12}) );

        ann.addLayer( new rna::Reshape({12*12}, true) );
        ann.addLayer( new rna::Linear(12*12, 10) );

//        ann.addLayer( new rna::Convolutional({1, 24, 24}, {5, 5}, 12) );
//        ann.addLayer( new rna::Tanh() );
//
//        ann.addLayer( new rna::Reshape({12*20*20}, true) );
//
//        ann.addLayer( new rna::Linear(12*20*20, 10) );
//        ann.addLayer( new rna::Tanh() );

//        Tensor inputB, outputB;
//        rna::randomMinibatch(dataSet, inputB, outputB, 100);
//
//
//
//        auto debut = GetTickCount();
//
//        ann.feedForward(inputB);
//
//        auto time = GetTickCount()-debut;
//        std::cout << "Temps: " << (time>1000?time/1000:time) << (time>1000?" s":" ms") << std::endl;


        ann.openCL(deviceType);
        Tensor output = ann.feedForward(dataSet[0].input);

        std::cout << output << std::endl;


        return 0;
    }

    if ("TEST2" == selection)
    {
        loadMNIST(10, dataSet);
        ann.loadFromFile("res/Networks/mnistGPU.rna");
//        ann.openCL(deviceType);

        Tensor i = dataSet[0].input;
        i.resize({1, 28, 28});

        auto debut = GetTickCount();


        std::cout << ann.feedForward(i) << std::endl;

        auto time = GetTickCount()-debut;
        std::cout << "Temps: " << (time>1000?time/1000:time) << (time>1000?" s":" ms") << std::endl;

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

    std::cout << "Loaded MNIST" << std::endl;
}

void validateMNIST(rna::Network& _ann, rna::DataSet& _dataSet)
{
    int correct = 0;

    for (auto& example: _dataSet)
    {
        Tensor output = _ann.feedForward(example.input);

        if (output.argmax()[0] == example.output(0))
            correct++;
    }

    std::cout << "Validation: " << correct << " over " << _dataSet.size() << " examples" << std::endl;
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

    rna::Optimizer<rna::MSE> op(0.001f, 0.9f);

    ann.train(op, dataSet, 5000, 5000, 32);

    for (unsigned i(0); i < 9; i++)
    {
        Tensor output = ann.feedForward(Vector({(Tensor::value_type)i})); output.round(2);
        std::cout << Vector({(Tensor::value_type)i}) << ": " << output << std::endl;
    }
}

