#pragma once

#include "Layer.h"

namespace rna
{

class Reshape: public Layer
{
    public:
        Reshape(coords_t _dimensions = {});
        Reshape(std::ifstream& _file);

        virtual const Tensor& feedForward(const Tensor& _input);
        virtual const Tensor& backprop(const Tensor& _input, const Tensor& _gradOutput);

        virtual void saveToFile(std::ofstream& _file) const;

    private:
        coords_t outputSize;
};

}
