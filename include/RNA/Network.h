#pragma once

#include <string>
#include <iostream>

#include "Layers/Layer.h"


namespace rna
{

class Network
{
    public:
        Network();
        Network(Network&& _network);
//        Network(const Network& _network);

        ~Network();

        Network& operator=(Network _network);

        void add(Layer* _layer);
        void clear();

        #ifdef USE_OPENCL
        void openCL(cl_device_type _deviceType = CL_DEVICE_TYPE_ALL);
        void releaseCL();

        const Tensor& feedForward(const Tensor& _input);
        void backprop(const Tensor& _input, const Tensor& _outputGrad);

        const Tensor& feedForward(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch);
        void backprop(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch);
        #else
        const Tensor& feedForward(const Tensor& _input);
        void backprop(const Tensor& _input, const Tensor& _outputGrad);
        #endif // USE_OPENCL


        #ifdef USE_OPENCL
        cl::Context& getContext();
        #endif // USE_OPENCL

        const Tensor& getOutput() const;
        Layer* getLayer(size_t _index) const;

        void getParams(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad) const;

        bool saveToFile(const std::string& _file) const;
        bool loadFromFile(const std::string& _file);

        friend void swap(Network& first, Network& second)
        {
            using std::swap;

            swap(first.layers, second.layers);

            // FIXME: swap contexts
//            #ifdef USE_OPENCL
//            swap(first.context, second.context);
//            #endif // USE_OPENCL
        }

    private:
        std::vector<Layer*> layers;

        #ifdef USE_OPENCL
        cl::Context context;
        #endif // USE_OPENCL
};

}
