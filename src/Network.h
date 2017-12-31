#pragma once

#include <string>

#include "Layers/Layer.h"
#include "Optimizer.h"
#include "clWrapper.h"


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

        void openCL(cl_device_type _deviceType = CL_DEVICE_TYPE_CPU);
        void releaseCL();

        // TODO: Should not copy return value
        Tensor feedForward(const Tensor& _input);
        Tensor feedForward(Tensor& _input);

        void backprop(const Tensor& _input, const Tensor& _gradOutput);
        void backprop(Tensor& _input, Tensor& _gradOutput);

        template<typename L>
        void train(Optimizer<L>& _optimizer, const DataSet& _dataSet, unsigned _maxEpochs, unsigned _epochsBetweenReports, unsigned _minibatchSize = 32);

//        template<typename L>
//        void QLearn(Optimizer<L>& _optimizer, Network& _target, const Memory& _memory, unsigned _miniBatchSize, double _discount);


        void zeroParametersGradients();
        void updateParameters(Tensor::value_type _learningRate, Tensor::value_type _inertia);

        bool saveToFile(const std::string& _file) const;
        bool loadFromFile(const std::string& _file);

    private:
        Tensor feedForwardCPU(const Tensor& _input);
        Tensor feedForwardCL(Tensor& _inputBatch);

        void backpropCPU(const Tensor& _input, const Tensor& _gradOutput);
        void backpropCL(Tensor& _inputBatch, Tensor& _gradOutputBatch);

        template<typename L>
        void trainCPU(Optimizer<L>& _optimizer, const DataSet& _dataSet, unsigned _maxEpochs, unsigned _epochsBetweenReports, unsigned _minibatchSize);

        template<typename L>
        void trainCL(Optimizer<L>& _optimizer, const DataSet& _dataSet, unsigned _maxEpochs, unsigned _epochsBetweenReports, unsigned _minibatchSize);

        cl::ContextWrapper context;

        std::vector<Layer*> layers;

        friend void swap(Network& first, Network& second)
        {
            using std::swap;

            swap(first.layers, second.layers);
        }
};

}

#include "Network.inl"