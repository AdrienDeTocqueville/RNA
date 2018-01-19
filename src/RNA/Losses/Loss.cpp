#include "RNA/Losses/Loss.h"

namespace rna
{

Loss::Loss()
{ }

Loss::~Loss()
{
    #ifdef USE_OPENCL
    releaseCL();
    #endif // USE_OPENCL
}

#ifdef USE_OPENCL
void Loss::releaseCL()
{
	gradientKernel.release();
}
#endif // USE_OPENCL

}
