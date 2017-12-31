#pragma once

#include <string>

std::string askFile(std::string _default, std::string _description = "Entrez un fichier");
std::string getExtension(std::string file);

double clamp(double _min, double _val, double _max);
