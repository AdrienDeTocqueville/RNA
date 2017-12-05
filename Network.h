#pragma once

#include <string>

#include "Tensor.h"

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

        void addLayer(Layer* _layer);

//        no matching function for call to 'rna::Network::add(<brace-enclosed initializer list>)'|
//        template<typename T, typename... Args>
//        void add(Args&&... args)
//        {
//            layers.push_back(new T(args...));
//        }

        Tensor feedForward(const Tensor& _input);
        void backprop(const Tensor& _input, const Tensor& _gradOutput);

        void train(LossFunction* _loss, const DataSet& _dataSet, double _learningRate, double _inertia, unsigned _maxEpochs, unsigned _epochsBetweenReports);
        void QLearn(LossFunction* _loss, Network& target, const Memory& _memory, double _learningRate, double _inertia, unsigned _miniBatchSize, double _discount);

        void zeroParametersGradients();
        void updateParameters(double _learningRate, double _inertia);

        void validate(const DataSet& _dataSet);

        bool saveToFile(std::string _file) const;
        bool loadFromFile(std::string _file);

    private:
        std::vector<Layer*> layers;

        friend void swap(Network& first, Network& second)
        {
            using std::swap;

            swap(first.layers, second.layers);
        }
};

}
