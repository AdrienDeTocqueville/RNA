#include "RNA.h"

#include <algorithm>

#include <iostream>
#include <fstream>
#include <ctime>



namespace rna
{

Network::Network()
{
//    {
//        #warning : Seed is not random
//        srand(2);
//    }
    srand(time(NULL));
}

Network::~Network()
{
    for (unsigned i(0) ; i < layers.size() ; ++i)
        delete layers[i];
}

void Network::addLayer(Layer* _layer)
{
    layers.push_back(_layer);
}

Tensor Network::feedForward(const Tensor& _input)
{
    if (!layers.size())
        return _input;

    Tensor output = layers.front()->feedForward(_input);

    for (unsigned i(1) ; i < layers.size() ; ++i)
        layers[i]->feedForward( layers[i-1]->output );

    return layers.back()->output;
}

void Network::backprop(const Tensor& _input, const Tensor& _gradOutput)
{
    Tensor gradient = _gradOutput;

    for (unsigned l(layers.size()-1) ; l >= 1 ; l--)
        gradient = layers[l]->backprop(layers[l-1]->output, gradient);

    layers[0]->backprop(_input, gradient);
}

void Network::zeroParametersGradients()
{
    for (unsigned l(0) ; l < layers.size() ; ++l)
        layers[l]->zeroParametersGradients();
}

void Network::train(LossFunction* _loss, const DataSet& _dataSet, double _learningRate, double _inertia, unsigned _maxEpochs, unsigned _epochsBetweenReports)
{
    static unsigned miniBatchSize = 20;

    unsigned step = 0;
    double error = 0.0;

    _learningRate = std::max(_learningRate, 0.0);
    _inertia = clamp(0.0, _inertia, 1.0);

    do
    {
        zeroParametersGradients();
        for (unsigned i(0) ; i < miniBatchSize ; ++i)
        {
//            auto& example = _dataSet[ iRand(0, _dataSet.size()-1) ];
            auto& example = _dataSet[ (step*miniBatchSize + i) % _dataSet.size() ];

            Tensor output = feedForward( example.input );


            error += _loss->getLoss(output, example.output);
            Tensor gradient = _loss->getGradient(output, example.output);


            backprop(example.input, gradient);
        }
        updateParameters(_learningRate, _inertia);


        ++step;
        if (step%_epochsBetweenReports == 0)
        {
            std::cout << "At step " << step << ":" << std::endl;
            std::cout << "Error = " << error/(_epochsBetweenReports*miniBatchSize) << std::endl;

            std::cout << std::endl;

            error = 0.0;
        }
    }
    while (step < _maxEpochs);

    delete _loss;

//    if (step == _maxEpochs)
//        std::cout << "Pas de convergence" << std::endl;
//    else
//        std::cout << "Convergence en: " << step << " iterations" << std::endl;
}

void Network::updateParameters(double _learningRate, double _inertia)
{
    for (unsigned l(0) ; l < layers.size() ; ++l)
        layers[l]->updateParameters(_learningRate, _inertia);
}

void Network::validate(const DataSet& _dataSet)
{
    double correct = 0.0;

    for (auto& example: _dataSet)
    {
        Tensor output = feedForward(example.input);

        if (output.argmax()[0] == example.output(0))
            correct++;
    }

    std::cout << "Validation: " << correct << " over " << _dataSet.size() << " examples" << std::endl;
}

bool Network::saveToFile(std::string _file) const
{
    std::ofstream file(_file);

    if (!file)
    {
        std::cout << "Network::saveToFile => Unable to create file: " << _file << std::endl;
        return false;
    }
    else
        std::cout << "Saving network to file: " << _file << std::endl;

    for (const Layer* layer: layers)
    {
        file << layer->type << std::endl;
        layer->saveToFile(file);

        file << std::endl << std::endl;
    }

    return true;
}

bool Network::loadFromFile(std::string _file)
{
    std::ifstream file(_file);

    if (!file)
    {
        std::cout << "Network::loadFromFile => File not found: " << _file << std::endl;
        return false;
    }
    else
        std::cout << "Loading network from file: " << _file << std::endl;

    Layer* layer = nullptr;
    std::string layerType;

    while (1)
    {
        file >> layerType;

        if (file.peek() == EOF)
            break;


        if ("Linear" == layerType)
            layer = new Linear(file);

        else if ("Reshape" == layerType)
            layer = new Reshape(file);

        else if ("Convolutional" == layerType)
            layer = new Convolutional(file);


        else if ("MaxPooling" == layerType)
            layer = new MaxPooling();

        else if ("Tanh" == layerType)
            layer = new Tanh();

        else if ("ReLU" == layerType)
            layer = new ReLU();

        else if ("LogSoftMax" == layerType)
            layer = new LogSoftMax();

        else
            std::cout << "Unknown layer type: " << layerType << std::endl;


        layers.push_back(layer);
    }

    return true;
}

}
