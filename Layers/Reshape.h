#pragma once

#include "Layer.h"

namespace rna
{

class Reshape: public Layer
{
    public:
        Reshape(coords_t _dimensions = {});
        Reshape(std::ifstream& _file);

        virtual void feedForwardCPU(const Tensor& _input);
        virtual void backpropCPU(const Tensor& _input, const Tensor& _gradOutput);

        virtual void saveToFile(std::ofstream& _file) const override;

    private:
        coords_t outputSize;
};

}
