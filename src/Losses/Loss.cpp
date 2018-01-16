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
	gradientKernel.release();
}

}
