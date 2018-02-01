#include "MNIST.h"

#include <iostream>
#include <fstream>

namespace MNIST
{

void test()
{
    std::cout << std::endl << "=== Testing MNIST ===" << std::endl;


    rna::DataSet training, testing, testBatch;
    load(training, testing);

    #ifdef USE_OPENCL
    rna::Network ann;
    ann.openCL(cl::DeviceType::GPU);

    ann.add( new rna::Reshape({28*28}) );
    ann.add( new rna::Linear(28*28, 10) );
    ann.add( new rna::LogSoftMax() );


    rna::Supervised trainer(ann);
        trainer.setLoss<rna::NLL>();
        trainer.setOptimizer<rna::SGD>(0.5f);

    trainer.train(training, 1000, 100);

    int correct = 0;
    for (rna::Example& batch: testing)
    {
        batch.input.resize({1, 28*28});
        const Tensor& output = ann.feedForward(batch.input);

        if (output.argmax()[1] == batch.output[0])
            correct++;
    }

    std::cout << "Correct = " << correct << " / " << testing.size() << std::endl;

    #else
    rna::Network ann;

    ann.add( new rna::Reshape({28*28}) );
    ann.add( new rna::Linear(28*28, 10) );
    ann.add( new rna::LogSoftMax() );


    rna::Supervised trainer(ann);
        trainer.setLoss<rna::NLL>();
        trainer.setOptimizer<rna::SGD>(0.5f);

    trainer.train(training, 1000, 100);

    int correct = 0;
    for (const rna::Example& batch: testing)
    {
        const Tensor& output = ann.feedForward(batch.input);

        if (output.argmax()[0] == batch.output[0])
            correct++;
    }

    std::cout << "Correct = " << correct << " / " << testing.size() << std::endl;
    #endif // USE_OPENCL
}

void load(rna::DataSet& _training, rna::DataSet& _testing)
{
    std::string baseDir = "res/MNIST/";

    LoadImages(_training, baseDir+ "train-images.idx3-ubyte");
    LoadLabels(_training, baseDir+ "train-labels.idx1-ubyte");

    LoadImages(_testing, baseDir+ "t10k-images.idx3-ubyte");
    LoadLabels(_testing, baseDir+ "t10k-labels.idx1-ubyte");
}


int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void LoadImages(rna::DataSet& _data, std::string _file)
{
    std::ifstream file (_file, std::ios::binary);

    if (file)
    {
        int magic_number, number_of_images;
        unsigned rows, columns;

        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);

        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);

        file.read((char*)&rows,sizeof(rows));
        rows = ReverseInt(rows);

        file.read((char*)&columns,sizeof(columns));
        columns = ReverseInt(columns);

        _data.resize(number_of_images);
        for (auto& example: _data)
        {
            example.input.resize({1, rows, columns});

            for (unsigned i(0) ; i < columns ; i++)
            {
                for (unsigned j(0) ; j < rows ; j++)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));

                    example.input(0, j, i) = (((int)temp)/255.0)*2.0 - 1.0;
                }
            }
        }
    }
    else
        std::cout << "Unable to open: " << _file << std::endl;
}

void LoadLabels(rna::DataSet& _data, std::string _file)
{
    std::ifstream file (_file, std::ios::binary);

    if (file)
    {
        int magic_number=0;
        int number_of_items=0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number= ReverseInt(magic_number);

        file.read((char*)&number_of_items, sizeof(number_of_items));
        number_of_items= ReverseInt(number_of_items);

        _data.resize(number_of_items);
        for (auto& example: _data)
        {
            example.output.resize({1});

            unsigned char temp=0;
            file.read((char*)&temp, sizeof(temp));
            example.output(0) = (Tensor::value_type)temp;
        }

    }
    else
        std::cout << "Unable to open: " << _file << std::endl;
}

}