#pragma once

#include <string>

#include "Layers/Layer.h"


namespace rna
{

struct Example
{
    Tensor input;
    Tensor output;
};

struct Transition
{
    Tensor state;
    size_t action;
    Tensor::value_type reward;
    Tensor nextState;

    bool terminal;
};

using DataSet = std::vector<Example>;
using Memory = std::vector<Transition>;

void randomMinibatch(const DataSet& _dataSet, Tensor& _inputBatch, Tensor& _outputBatch, const unsigned& _minibatchSize);

class Network
{
    public:
        Network();
        Network(Network&& _network);
//        Network(const Network& _network);

        ~Network();

        Network& operator=(Network _network);

        void addLayer(Layer* _layer); // Warning: don't send the same pointer twice
        Layer* getLayer(size_t _index); // temp

//        no matching function for call to 'rna::Network::add(<brace-enclosed initializer list>)'|
//        template<typename T, typename... Args>
//        void add(Args&&... args)
//        {
//            layers.push_back(new T(args...));
//        }

        void openCL(cl_device_type _deviceType = CL_DEVICE_TYPE_ALL);
        void releaseCL();

        const Tensor& feedForward(const Tensor& _input);
        void backprop(const Tensor& _input, const Tensor& _gradOutput);

        const Tensor& feedForwardCPU(const Tensor& _input);
        const Tensor& feedForwardCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch);

        void backpropCPU(const Tensor& _input, const Tensor& _gradOutput);
        void backpropCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch);


        cl::Context& getContext();
        void getParams(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad) const;

        bool saveToFile(const std::string& _file) const;
        bool loadFromFile(const std::string& _file);

        friend void swap(Network& first, Network& second)
        {
            using std::swap;

            swap(first.layers, second.layers);
        }

    private:
        cl::Context context;

        std::vector<Layer*> layers;
};

}
