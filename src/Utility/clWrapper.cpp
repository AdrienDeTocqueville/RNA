#include "clWrapper.h"

#include <cstdlib>
#include <fstream>
#include <iostream>

#include "Error.h"
#include "Tensor.h"

namespace cl
{

std::vector<cl_platform_id> getPlatformsIds()
{
    std::vector<cl_platform_id> ids;

    cl_int error;

    cl_uint platCount;
    error = clGetPlatformIDs(0, nullptr, &platCount);
    if (error != CL_SUCCESS)
        Error::add(ErrorType::WARNING, "OpenCL: Failed to get platform count with error code " + toString(error));

    ids.resize(platCount);
    error = clGetPlatformIDs(platCount, ids.data(), nullptr);
    if (error != CL_SUCCESS)
        Error::add(ErrorType::WARNING, "OpenCL: Failed to get platforms with error code " + toString(error));

    return ids;
}

std::vector<cl_device_id> getDeviceIds(cl_device_type _deviceType, cl_platform_id _platformId)
{
    std::vector<cl_device_id> ids;

    cl_int error;

    cl_uint deviceCount = 0;
    error = clGetDeviceIDs(_platformId, _deviceType, 0, nullptr, &deviceCount);
    if (error != CL_SUCCESS)
        Error::add(ErrorType::WARNING, "OpenCL: Failed to get devices count with error code " + toString(error));

    ids.resize(deviceCount);
    error = clGetDeviceIDs(_platformId, _deviceType, deviceCount, ids.data(), nullptr);
    if (error != CL_SUCCESS)
        Error::add(ErrorType::WARNING, "OpenCL: Failed to get devices with error code " + toString(error));

    return ids;
}

std::string getVersion(cl_platform_id _platformId)
{
    cl_uint charCount = 0;
    clGetPlatformInfo(_platformId, CL_PLATFORM_VERSION, 0, nullptr, &charCount);

    char* cversion = new char[charCount];
    clGetPlatformInfo(_platformId, CL_PLATFORM_VERSION, charCount, cversion, nullptr);

    std::string version(cversion);
    delete[] cversion;

    return version;
}

Context::Context():
    deviceId(0)
{ }

Context::Context(const Context& c):
    Wrapper<cl_context>()
{
    create(c.getDeviceId());
}

void Context::create(cl_device_type _deviceType)
{
    cl_platform_id platformIds = getPlatformsIds().front();
    deviceId = getDeviceIds(_deviceType, platformIds).front();

    std::cout << "OpenCL version: " << getVersion(platformIds) << std::endl;

    create(deviceId);
}

void Context::create(cl_device_id _deviceId)
{
    if (id || !_deviceId)
        return;

    cl_int error;

    deviceId = _deviceId;

    id = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &error);
    if (error != CL_SUCCESS || id == nullptr)
        Error::add(ErrorType::WARNING, "OpenCL: Failed to create id with error code " + toString(error));
}

void Context::release()
{
    clReleaseContext(id);
    id = 0;
    deviceId = 0;

    programs.clear();
}

const Program& Context::getProgram(const std::string& _path)
{
    Program& program = programs[_path];
    program.create(*this, _path);

    return program;
}

cl_device_id Context::getDeviceId() const
{
    return deviceId;
}


/// Program
std::string Program::baseDirectory = "";

void Program::create(const Context& _context, const std::string& _path)
{
    if (id)
        return;

    cl_int error;
    std::string path = baseDirectory + _path;

	std::ifstream in(_path);
	if (!in)
        return Error::add(ErrorType::FILE_NOT_FOUND, _path);

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

void Program::release()
{
	clReleaseProgram(id);
    id = 0;
}

void Program::setBaseDirectory(const std::string& _baseDirectory)
{
    baseDirectory = _baseDirectory;

    if (baseDirectory.size() && baseDirectory.back() != '/')
        baseDirectory.append("/");
}


/// Kernel
void Kernel::create(const Program& _program, const std::string& _name)
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

void Kernel::release()
{
	clReleaseKernel(id);
    id = 0;
}

void Kernel::setArg(cl_uint _index, size_t _size, const void* _value)
{
    cl_int error = clSetKernelArg(id, _index, _size, _value);

    if (error != CL_SUCCESS)
        Error::add(ErrorType::UNKNOWN_ERROR, "Kernel arg number " + toString(_index) + " error: " + toString(error));
}

void Kernel::setArg(cl_uint _index, const Tensor& _value)
{
    setArg(_index, sizeof(cl_mem), &_value.getBuffer());
}


/// CommandQueue
void CommandQueue::create(const Context& _context, bool _inOrder)
{
    if (id)
        return;

    context = &_context;
    inOrder = _inOrder;

    cl_int error;

    // About out of order mode: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateCommandQueue.html
    cl_command_queue_properties cqProperties = _inOrder? 0: CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, cqProperties, 0};

    id = clCreateCommandQueueWithProperties(_context(), _context.getDeviceId(), properties, &error);
//    id = clCreateCommandQueue(_context(), _context.getDeviceId(), cqProperties, &error);

    if (error != CL_SUCCESS)
        Error::add(ErrorType::UNKNOWN_ERROR, "clCreateCommandQueue() error: " + toString(error));
}

void CommandQueue::release()
{
    clReleaseCommandQueue(id);
    id = 0;

    context = nullptr;
}

void CommandQueue::join() const
{
    clFinish(id);
}

const Context& CommandQueue::getContext() const
{
    return *context;
}

void CommandQueue::enqueueKernel(Kernel& _kernel, const coords_t& _globalWorkSize, cl_event* _event)
{
	cl_int error = clEnqueueNDRangeKernel(id, _kernel(), _globalWorkSize.size(), nullptr, _globalWorkSize.data(), nullptr, 0, nullptr, _event);

	if (error != CL_SUCCESS)
        Error::add(ErrorType::USER_ERROR, "Enqueue kernel: " + toString(error));
}

void CommandQueue::enqueueBarrier(const std::vector<cl_event>& _events)
{
    cl_int error = clEnqueueBarrierWithWaitList(id, _events.size(), _events.data(), nullptr);

	if (error != CL_SUCCESS)
        Error::add(ErrorType::USER_ERROR, "Enqueue barrier: " + toString(error));
}

void CommandQueue::enqueueBarrier()
{
    cl_int error = clEnqueueBarrierWithWaitList(id, 0, nullptr, nullptr);

	if (error != CL_SUCCESS)
        Error::add(ErrorType::USER_ERROR, "Enqueue barrier: " + toString(error));
}

}
