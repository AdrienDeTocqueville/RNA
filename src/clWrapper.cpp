#include "clWrapper.h"

#include <fstream>
#include <iostream>

#include "Utility/Error.h"
#include "Utility/util.h"

namespace rna
{

std::map<std::string, clProgramWrapper> clProgramWrapper::programs;

clProgramWrapper::clProgramWrapper():
    id(0)
{ }

clProgramWrapper::~clProgramWrapper()
{
	clReleaseProgram(id);
}

cl_program clProgramWrapper::get(cl_context _context, cl_device_id _deviceId, const std::string& _program)
{
    clProgramWrapper& p = programs[_program];
    if (p.id == 0)
        p.load(_context, _deviceId, _program);

    return p.id;
}

void clProgramWrapper::load(cl_context _context, cl_device_id _deviceId, const std::string& _program)
{
    cl_int error;

	std::ifstream in(_program);
	if (!in)
        Error::add(ErrorType::FILE_NOT_FOUND, _program);

	std::string src( (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>() );
    const char* strings = src.c_str();
    const size_t length = src.size();


    id = clCreateProgramWithSource(_context, 1, &strings, &length, &error);
    if (error != CL_SUCCESS)
        Error::add(ErrorType::UNKNOWN_ERROR, "clCreateProgramWithSource() error: " + toString(error));


    error = clBuildProgram(id, 1, &_deviceId, nullptr, nullptr, nullptr);//"-cl-std=CL2.0"
    if (error != CL_SUCCESS)
    {
        size_t logLength;
        error = clGetProgramBuildInfo(id, _deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLength);

        char* log = new char[logLength];
        error = clGetProgramBuildInfo(id, _deviceId, CL_PROGRAM_BUILD_LOG, logLength, log, nullptr);

        std::cout << log << std::endl;
        delete[] log;

        Error::add(ErrorType::USER_ERROR, "clBuildProgram() error");
    }
}

cl_kernel loadKernel(const cl_context& _context, const cl_device_id& _deviceId, const std::string& _program, const std::string& _kernel)
{
    cl_int error;
    cl_program program = clProgramWrapper::get(_context, _deviceId, _program);

    cl_kernel kernel = clCreateKernel(program, _kernel.c_str(), &error);
    if (error != CL_SUCCESS)
    {
        if (error == CL_INVALID_KERNEL_NAME)
            Error::add(ErrorType::USER_ERROR, "Invalid kernel name: " + _kernel);
        else
            Error::add(ErrorType::UNKNOWN_ERROR, "clCreateKernel() error: " + toString(error));
    }

    return kernel;
}

void execKernel(const cl_command_queue& _commandQueue, const cl_kernel& _kernel, const coords_t& _globalWorkSize)
{
	cl_int error = clEnqueueNDRangeKernel(_commandQueue, _kernel, _globalWorkSize.size(), nullptr, _globalWorkSize.data(), nullptr, 0, nullptr, nullptr);

	if (error != CL_SUCCESS)
    {
        Error::add(ErrorType::USER_ERROR, "execKernel: " + toString(error));
    }
}

void execKernel(const cl_command_queue& _commandQueue, const cl_kernel& _kernel, const coords_t& _globalWorkSize, const coords_t& _globalWorkOffset)
{
	cl_int error = clEnqueueNDRangeKernel(_commandQueue, _kernel, _globalWorkSize.size(), _globalWorkOffset.data(), _globalWorkSize.data(), nullptr, 0, nullptr, nullptr);

	if (error != CL_SUCCESS)
    {
        Error::add(ErrorType::USER_ERROR, "execKernel: " + toString(error));
    }
}

}
