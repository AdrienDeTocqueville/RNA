#include "clWrapper.h"

#include <cstdlib>
#include <fstream>
#include <iostream>

#include "Utility/Error.h"

namespace rna
{
namespace cl
{

ContextWrapper::ContextWrapper():
    deviceId(0)
{ }

ContextWrapper::ContextWrapper(const ContextWrapper& c):
    Wrapper<cl_context>()
{
    create(c.getDeviceId());
}

void ContextWrapper::create(cl_device_type _deviceType)
{
    cl_int error;

    cl_platform_id platform_id;
    error = clGetPlatformIDs(1, &platform_id, nullptr);
    if (error != CL_SUCCESS)
        Error::add(ErrorType::WARNING, "OpenCL: Failed to get platforms with error code " + toString(error));

    error = clGetDeviceIDs(platform_id, _deviceType, 1, &deviceId, nullptr);
    if (error != CL_SUCCESS)
        Error::add(ErrorType::WARNING, "OpenCL: Failed to get devices with error code " + toString(error));

    // Print version
    if (true)
    {
        cl_uint deviceIdCount = 0;
        clGetDeviceInfo(deviceId, CL_DEVICE_VERSION, 0, nullptr, &deviceIdCount);

        char* v = (char*)malloc(deviceIdCount*sizeof(char));
        clGetDeviceInfo(deviceId, CL_DEVICE_VERSION, deviceIdCount, v, nullptr);

        std::cout << "OpenCL version: " << v << std::endl;

        free(v);
    }

    create(deviceId);
}

void ContextWrapper::create(cl_device_id _deviceId)
{
    if (id || !_deviceId)
        return;

    cl_int error;

    deviceId = _deviceId;

    id = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &error);
    if (error != CL_SUCCESS || id == nullptr)
        Error::add(ErrorType::WARNING, "OpenCL: Failed to create id with error code " + toString(error));
}

void ContextWrapper::release()
{
    clReleaseContext(id);
    id = 0;
    deviceId = 0;

    programs.clear();
}

const ProgramWrapper& ContextWrapper::getProgram(const std::string& _path)
{
//    auto element = programs.emplace(std::piecewise_construct, std::make_tuple(_path), std::make_tuple(*this, _path));
//
//    return &element.first->second;
    ProgramWrapper& program = programs[_path];
    program.create(*this, _path);

    return program;
}

cl_device_id ContextWrapper::getDeviceId() const
{
    return deviceId;
}


/// ProgramWrapper
void ProgramWrapper::create(const ContextWrapper& _context, const std::string& _path)
{
    if (id)
        return;

    cl_int error;

	std::ifstream in(_path);
	if (!in)
        Error::add(ErrorType::FILE_NOT_FOUND, _path);

    // Read file
	std::string src( (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>() );
    const char* strings = src.c_str();
    const size_t length = src.size();

    // Create Program
    id = clCreateProgramWithSource(_context(), 1, &strings, &length, &error);
    if (error != CL_SUCCESS)
        Error::add(ErrorType::UNKNOWN_ERROR, "Error " + toString(error) + " while creating program: " + _path);

    // Compile program
    auto deviceId = _context.getDeviceId();
    error = clBuildProgram(id, 1, &deviceId, nullptr, nullptr, nullptr);//"-cl-std=CL2.0"

    if (error == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t logLength;
        error = clGetProgramBuildInfo(id, deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLength);

        char* log = new char[logLength];
        error = clGetProgramBuildInfo(id, deviceId, CL_PROGRAM_BUILD_LOG, logLength, log, nullptr);

        std::cout << log << std::endl;
        delete[] log;

        Error::add(ErrorType::USER_ERROR, "Build failed for: " + _path);
    }
    else if (error != CL_SUCCESS)
        Error::add(ErrorType::UNKNOWN_ERROR, "Error " + toString(error) + " while building program: " + _path);
}

void ProgramWrapper::release()
{
	clReleaseProgram(id);
    id = 0;
}


/// KernelWrapper
void KernelWrapper::create(const ProgramWrapper& _program, const std::string& _name)
{
    if (id)
        return;

    cl_int error;

    id = clCreateKernel(_program(), _name.c_str(), &error);

    if (error != CL_SUCCESS)
    {
        if (error == CL_INVALID_KERNEL_NAME)
            Error::add(ErrorType::USER_ERROR, "Invalid kernel name: " + _name);
        else
            Error::add(ErrorType::UNKNOWN_ERROR, "clCreateKernel() error: " + toString(error));
    }
}

void KernelWrapper::release()
{
	clReleaseKernel(id);
    id = 0;
}

void KernelWrapper::setArg(cl_uint _index, size_t _size, const void* _value)
{
    clSetKernelArg(id, _index, _size, _value);
}

void KernelWrapper::setArg(cl_uint _index, const Tensor& _value)
{
    clSetKernelArg(id, _index, sizeof(cl_mem), &_value.getBuffer());
}

void KernelWrapper::setArg(cl_uint _index, int _value)
{
    clSetKernelArg(id, _index, sizeof(int), &_value);
}

void KernelWrapper::enqueue(const cl_command_queue& _commandQueue, const coords_t& _globalWorkSize)
{
	cl_int error = clEnqueueNDRangeKernel(_commandQueue, id, _globalWorkSize.size(), nullptr, _globalWorkSize.data(), nullptr, 0, nullptr, nullptr);

	if (error != CL_SUCCESS)
    {
        Error::add(ErrorType::USER_ERROR, "Kernel enqueue: " + toString(error));
    }
}

}
}
