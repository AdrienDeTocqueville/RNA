#include "RNA/Layers/Layer.h"

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
    #ifdef USE_OPENCL
    releaseCL();
    #endif // USE_OPENCL
}

#ifdef USE_OPENCL
void Layer::releaseCL()
{
	forwardKernel.release();
	backwardKernel.release();
}
#endif // USE_OPENCL

const Tensor& Layer::getOutput() const
{
    return output;
}

const Tensor& Layer::getInputGrad() const
{
    return inputGrad;
}

const std::string& Layer::getType() const
{
    return type;
}

void Layer::saveToFile(std::ofstream& _file) const
{
    _file << type << std::endl;
}

}
