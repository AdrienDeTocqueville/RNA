#pragma once

#include <CL/opencl.h>
#include <string>
#include <map>

#include "Utility/Tensor.h"

namespace rna
{

class clProgramWrapper
{
    public:
        clProgramWrapper();
        ~clProgramWrapper();

        static cl_program get(cl_context _context, cl_device_id _deviceId, const std::string& _program);

    private:
        static std::map<std::string, clProgramWrapper> programs;

        void load(cl_context _context, cl_device_id _deviceId, const std::string& _program);

        cl_program id;
};

cl_kernel loadKernel(const cl_context& _context, const cl_device_id& _deviceId, const std::string& _program, const std::string& _kernel);
void execKernel(const cl_command_queue& _commandQueue, const cl_kernel& _kernel, const coords_t& _globalWorkSize);

}
