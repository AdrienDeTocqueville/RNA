#include "Layer.h"

#include <fstream>

namespace rna
{

Tensor::value_type Layer::WEIGHT_INIT_MIN = -0.5f;
Tensor::value_type Layer::WEIGHT_INIT_MAX = 0.5f;

Tensor::value_type Layer::BIAS_INIT_MIN = 0.25f;
Tensor::value_type Layer::BIAS_INIT_MAX = 0.25f;

Layer::Layer(std::string _type):
    type(_type)
{ }

Layer::~Layer()
{
    releaseCL();
}

const Tensor& Layer::getOutput() const
{
    return output;
}

const Tensor& Layer::getInputGrad() const
{
    return inputGrad;
}

void Layer::saveToFile(std::ofstream& _file) const
{
    _file << type << std::endl;
}

void Layer::releaseCL()
{
	forwardKernel.release();
	backwardKernel.release();
}

}
