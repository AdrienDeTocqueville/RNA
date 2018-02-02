#include <iostream>
#include <ctime>

#include "windows.h"

#include "RL.h"
#include "MNIST.h"
#include "Utility/Random.h"

//#define RANDOM_SEED

void loadXOR(unsigned _size, rna::DataSet& _data);

int main()
{
    #ifdef RANDOM_SEED
        std::cout << "Seed: " << Random::getSeed() << std::endl << std::endl;
    #else
        Random::setSeed(1);

        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        SetConsoleTextAttribute(hConsole, 12);
        std::cout << "Warning: seed is fixed to " << Random::getSeed() << std::endl << std::endl;
        SetConsoleTextAttribute(hConsole, 7);
    #endif


    MNIST::test();

    RL::test();

    return 0;
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

