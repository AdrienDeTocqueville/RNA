#include "LossFunction.h"

namespace rna
{

LossFunction::LossFunction()
{ }

LossFunction::~LossFunction()
{
    releaseCL();
}

void LossFunction::releaseCL()
{
	lossKernel.release();
	gradientKernel.release();
}

}
