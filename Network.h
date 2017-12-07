#pragma once

#include <CL/opencl.h>
#include <string>

#include "Utility/Tensor.h"

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
    double reward;
    Tensor nextState;

    bool terminal;
};

using DataSet = std::vector<Example>;
using Memory = std::vector<Transition>;

class Layer;
class LossFunction;

class Network
{
    public:
        Network();
        Network(Network&& _network);
        Network(const Network& _network);

        ~Network();

        Network& operator=(Network _network);

        void addLayer(Layer* _layer); // Warning: don't send the same pointer twice

//        no matching function for call to 'rna::Network::add(<brace-enclosed initializer list>)'|
//        template<typename T, typename... Args>
//        void add(Args&&... args)
//        {
//            layers.push_back(new T(args...));
//        }

        void toGPU();

        Tensor feedForward(const Tensor& _input);
        void backprop(const Tensor& _input, const Tensor& _gradOutput);

        virtual Tensor GPUfeedForward(const Tensor& _inputBatch);

        void train(LossFunction* _loss, const DataSet& _dataSet, Tensor::value_type _learningRate, Tensor::value_type _inertia, unsigned _maxEpochs, unsigned _epochsBetweenReports);
        void GPUtrain(LossFunction* _loss, const DataSet& _dataSet, Tensor::value_type _learningRate, Tensor::value_type _inertia, unsigned _maxEpochs, unsigned _epochsBetweenReports, unsigned _minibatchSize);
        void QLearn(LossFunction* _loss, Network& target, const Memory& _memory, Tensor::value_type _learningRate, Tensor::value_type _inertia, unsigned _miniBatchSize, double _discount);

        void zeroParametersGradients();
        void updateParameters(Tensor::value_type _learningRate, Tensor::value_type _inertia);

        void validate(const DataSet& _dataSet);

        bool saveToFile(std::string _file) const;
        bool loadFromFile(std::string _file);

    private:
        cl_context context;
        cl_device_id deviceId;

        std::vector<Layer*> layers;

        friend void swap(Network& first, Network& second)
        {
            using std::swap;

            swap(first.layers, second.layers);
        }
};

}
