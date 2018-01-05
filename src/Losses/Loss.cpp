#include "Loss.h"

namespace rna
{

Loss::Loss()
{ }

Loss::~Loss()
{
    releaseCL();
}

void Loss::releaseCL()
{
	lossKernel.release();
	gradientKernel.release();
}

}
