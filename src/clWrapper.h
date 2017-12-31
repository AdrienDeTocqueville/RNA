#pragma once

#include <CL/opencl.h>
#include <string>
#include <map>

#include "Utility/Tensor.h"

namespace rna
{
namespace cl
{

class KernelWrapper;
class ProgramWrapper;

template <typename T>
class Wrapper
{
    public:
        Wrapper(): id(0) {}
        ~Wrapper() {}

        Wrapper(const Wrapper&& c) = delete;
        operator=(const Wrapper& c) = delete;

        virtual void release() = 0;

        T& operator()() { return id; }
        const T& operator()() const { return id; }
        operator bool () { return id != 0; }

    protected:
        T id;
};

class ContextWrapper: public Wrapper<cl_context>
{
    public:
        ContextWrapper();
        ContextWrapper(const ContextWrapper& c);

        void create(cl_device_type _deviceType);
        void create(cl_device_id _deviceId);
        void release();

        const ProgramWrapper& getProgram(const std::string& _path);

        cl_device_id getDeviceId() const;

    private:
        cl_device_id deviceId;

        std::map<std::string, ProgramWrapper> programs;
};

class ProgramWrapper: public Wrapper<cl_program>
{
    public:
        ProgramWrapper() {}
        ProgramWrapper(const ProgramWrapper& c) = delete;

        void create(const ContextWrapper& _context, const std::string& _path);
        void release();
};

class KernelWrapper: public Wrapper<cl_kernel>
{
    public:
        KernelWrapper() {}
        KernelWrapper(const KernelWrapper& c) = delete;

        void create(const ProgramWrapper& _program, const std::string& _name);
        void release();

        void setArg(cl_uint _index, size_t _size, const void* _value);
        void setArg(cl_uint _index, const Tensor& _value);
        void setArg(cl_uint _index, int _value);

        void enqueue(const cl_command_queue& _commandQueue, const coords_t& _globalWorkSize);
};

}
}
