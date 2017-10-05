#pragma once

#include <string>
#include <sstream>

template <typename T>
std::string toString(T _number)
{
    std::stringstream os;
    os << _number;

    return os.str();
}

std::string askFile(std::string _default, std::string _description = "Entrez un fichier");
std::string getExtension(std::string file);

double clamp(double _min, double _val, double _max);

int iRand(int iMin, int iMax);
double dRand(double dMin, double dMax);

double sigmoid(double _x);

double dSigmoid(double _x);
double dtanh(double _x);
