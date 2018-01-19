#pragma once

#include <CL/opencl.h>
#include <string>
#include <vector>
#include <map>

class Tensor;
using coords_t = std::vector<size_t>;

namespace cl
{

std::vector<cl_platform_id> getPlatformsIds();
std::vector<cl_device_id> getDeviceIds(cl_device_type _deviceType, cl_platform_id _platformId);

std::string getVersion(cl_platform_id _platformId);

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

class Program;
class Context: public Wrapper<cl_context>
{
    public:
        Context();
        Context(const Context& c);

        void create(cl_device_type _deviceType);
        void create(cl_device_id _deviceId);
        void release();

        const Program& getProgram(const std::string& _path);

        cl_device_id getDeviceId() const;

    private:
        cl_device_id deviceId;

        std::map<std::string, Program> programs;
};

class Program: public Wrapper<cl_program>
{
    public:
        Program() {}
        Program(const Program& c) = delete;

        void create(const Context& _context, const std::string& _path);
        void release();

        std::string getBuildLog(cl_device_id _deviceId) const;

        static void setBaseDirectory(const std::string& _baseDirectory = "");

    private:
        static std::string baseDirectory;
};

class Kernel: public Wrapper<cl_kernel>
{
    public:
        Kernel() {}
        Kernel(const Kernel& c) = delete;

        void create(const Program& _program, const std::string& _name);
        void release();

        void setArg(cl_uint _index, size_t _size, const void* _value);
        void setArg(cl_uint _index, const Tensor& _value);

        template <typename T>
        void setArg(cl_uint _index, T _value)
        { setArg(_index, sizeof(T), &_value); }
};

class CommandQueue: public Wrapper<cl_command_queue>
{
    public:
        CommandQueue() {}
        CommandQueue(const Context& _context, bool _inOrder = true) { create(_context, _inOrder); }
        CommandQueue(const CommandQueue& c) = delete;

        void create(const Context& _context, bool _inOrder = true);
        void release();

        void join() const;
        const Context& getContext() const;

        void enqueueKernel(Kernel& _kernel, const coords_t& _globalWorkSize, cl_event* _event = nullptr);
        void enqueueBarrier(const std::vector<cl_event>& _events);
        void enqueueBarrier();

    private:
        const Context* context;
        bool inOrder;
};

}
