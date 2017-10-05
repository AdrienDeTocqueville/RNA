#include "Util.h"

#include <iostream>
#include <cstdlib>
#include <cmath>

std::string askFile(std::string _default, std::string _description)
{
    std::cin.ignore();

    std::string input;

    std::cout << std::endl << "Fichier par default: " << _default << std::endl;
    std::cout << _description << ": ";   std::getline (std::cin, input);

    if (input.empty())  return _default;

    return input;
}

std::string getExtension(std::string file)
{
    size_t pos = file.rfind(".");
    return file.substr(pos+1, file.size()-1);
}

double clamp(double _min, double _val, double _max)
{
    return std::min(std::max(_min, _val), _max);
}

double dRand(double dMin, double dMax)
{
    double d = (double)rand() / RAND_MAX;
    return dMin + d * (dMax - dMin);
}

int iRand(int iMin, int iMax)
{
    return iMin + (rand() % (int)(iMax - iMin + 1));
}

double sigmoid(double _x)
{
    return 1.0 / ( 1.0 + exp(-_x) );
}

double dSigmoid(double _x)
{
    double s = sigmoid(_x);
    return s*(1.0 - s);
}

double dtanh(double _x)
{
    float t = tanh(_x);
    return 1.0 - t*t;
}
