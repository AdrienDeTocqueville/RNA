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
