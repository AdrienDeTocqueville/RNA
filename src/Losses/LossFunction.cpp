#include "LossFunction.h"

namespace rna
{

LossFunction::LossFunction()
{
    lossKernel = 0;
    gradientKernel = 0;
}

LossFunction::~LossFunction()
{
    clReleaseKernel(lossKernel);
    clReleaseKernel(gradientKernel);
}

}
