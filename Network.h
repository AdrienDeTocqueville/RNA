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

using DataSet = std::vector<Example>;

class Layer;
class LossFunction;

class Network
{
    public:
        Network();
        ~Network();

        void addLayer(Layer* _layer);

        Tensor feedForward(const Tensor& _input);
        void backprop(const Tensor& _input, const Tensor& _gradOutput);

        void zeroParametersGradients();
        void train(LossFunction* _loss, const DataSet& _dataSet, double _learningRate, double _inertia, unsigned _maxEpochs, unsigned _epochsBetweenReports);
        void updateParameters(double _learningRate, double _inertia);

        void validate(const DataSet& _dataSet);

        bool saveToFile(std::string _file) const;
        bool loadFromFile(std::string _file);

    private:
        std::vector<Layer*> layers;
};

}
