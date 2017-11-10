#include <iostream>
#include <windows.h>

#include "RNA.h"
#include "Image.h"

void loadXOR(unsigned _size, rna::DataSet& _data);
void loadMNIST(unsigned _size, rna::DataSet& _data);

int main()
{
    std::string selection = "conv";
    std::cout << "Select dataset (XOR, MNIST, CONV, TEST): ";
    std::cin >> selection;
    std::cout << std::endl;

    for (auto & c: selection)
        c = toupper(c);



    rna::DataSet dataSet;
    rna::Network ann;

    if ("n" == selection)
    {
        loadMNIST(1000, dataSet);

        ann.loadFromFile("Networks/mnist.rna");
    }

    if ("XOR" == selection)
    {
        loadXOR(100, dataSet);

        unsigned HU = 3;
        ann.addLayer( new rna::Linear(2, HU) );
        ann.addLayer( new rna::ReLU() );
        ann.addLayer( new rna::Linear(HU, 1) );
        ann.addLayer( new rna::Tanh() );

        ann.train(new rna::MSE(), dataSet, 0.001, 0.5, 5000, 500);
        ann.saveToFile("Networks/xor.rna");

        std::cout << ann.feedForward(Vector({-0.5, -0.5})) << std::endl; // -1.0
        std::cout << ann.feedForward(Vector({-0.5, 0.5})) << std::endl; // 1.0
        std::cout << ann.feedForward(Vector({0.5, -0.5})) << std::endl; // 1.0
        std::cout << ann.feedForward(Vector({0.5, 0.5})) << std::endl << std::endl; // -1.0

        return 0;
    }

    if ("MNIST" == selection)
    {
        loadMNIST(3000, dataSet);

        for (unsigned i(0) ; i < 2 ; i++)
        {
            std::cout << dataSet[i].output << std::endl;
            displayImage(dataSet[i].input, "Image", 8);
        }

        ann.addLayer( new rna::Reshape({28*28}) );
        ann.addLayer( new rna::Linear(28*28, 300) );
        ann.addLayer( new rna::Tanh() );
        ann.addLayer( new rna::Linear(300, 10) );
        ann.addLayer( new rna::Tanh() );
        ann.addLayer( new rna::LogSoftMax() );


        ann.train(new rna::NLL(), dataSet, 0.01, 0.0, 10000, 100);
        ann.validate(dataSet);
        ann.saveToFile("Networks/mnist1.rna");

        ann.train(new rna::NLL(), dataSet, 0.0005, 0.0, 50000, 500);
        ann.validate(dataSet);
        ann.saveToFile("Networks/mnist2.rna");
    }

    if ("CONV" == selection)
    {
        loadMNIST(1000, dataSet);
        for (rna::Example& e: dataSet)
            e.input.resize({1, 28, 28});

        std::cout << "DB loaded" << std::endl;

        {
            ann.addLayer( new rna::Convolutional({5, 5}, {1, 28, 28}, 4) );
            ann.addLayer( new rna::MaxPooling() );
            ann.addLayer( new rna::Tanh() );

            ann.addLayer( new rna::Convolutional({5, 5}, {4, 12, 12}, 12) );
            ann.addLayer( new rna::MaxPooling() );
            ann.addLayer( new rna::Tanh() );

            ann.addLayer( new rna::Reshape({12*4*4}) );

            ann.addLayer( new rna::Linear(12*4*4, 500) );
            ann.addLayer( new rna::Tanh() );

            ann.addLayer( new rna::Linear(500, 10) );
            ann.addLayer( new rna::Tanh() );

            ann.addLayer( new rna::LogSoftMax() );
        }

        std::cout << "ANN created" << std::endl;

        ann.train(new rna::NLL(), dataSet, 0.01, 0.3, 10000, 100);

        ann.saveToFile("Networks/leNet1.rna");
    }

    if ("TEST" == selection)
    {
        loadMNIST(1000, dataSet);

        ann.loadFromFile("Networks/leNet1.rna");

        for (unsigned i(0) ; i < 1000 ; i++)
        {
            dataSet[i].input.resize({1, 28, 28});

            std::cout << std::endl << std::endl << i << std::endl;
            std::cout << ann.feedForward(dataSet[i].input) << std::endl;
            std::cout << ann.feedForward(dataSet[i].input).argmax() << std::endl;
            std::cout << dataSet[i].output << std::endl;

            displayImage(dataSet[i].input, "Output", 8);
        }

        return 0;
    }

    return 0;
}

void loadXOR(unsigned _size, rna::DataSet& _data)
{
    _data.clear();
    _data.resize(_size);

    for (unsigned i(0) ; i < _data.size() ; i++)
    {
        Tensor input{2};
        input.randomize(-1.0, 1.0);

        Tensor output{1};
        if (input(0) * input(1) > 0.0)
            output(0) = -1.0;
        else
            output(0) = 1.0;

        _data[i] = {input, output};
    }
}

void loadMNIST(unsigned _size, rna::DataSet& _data) // 60000 examples
{
    _data.clear();
    _data.resize(_size);

    LoadMNISTImages(_data);
    LoadMNISTLabels(_data);
}
