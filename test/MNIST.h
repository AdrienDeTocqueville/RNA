#pragma once

#include "RNA/RNA.h"

namespace MNIST
{

void test();

void load(rna::DataSet& _training, rna::DataSet& _testing);

int ReverseInt (int i);
void LoadImages(rna::DataSet& _data, std::string _file);
void LoadLabels(rna::DataSet& _data, std::string _file);

}
