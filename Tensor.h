#pragma once

#include <initializer_list>
#include <vector>

#include "util.h"

using value_type = double;
using coords_t = std::vector<size_t>;

std::ostream& operator<<(std::ostream& os, const coords_t& coords);
coords_t operator/(const coords_t& coords, const int& a);

class Tensor
{
    public:
        static Tensor identityMatrix(size_t _dimension);

        Tensor();
        Tensor(const Tensor& _tensor);

        Tensor(const coords_t&  _dimensions);
        Tensor(const coords_t&  _dimensions, const value_type& _value);

        Tensor(std::initializer_list<size_t> _dimensions);
        Tensor(std::initializer_list<size_t> _dimensions, const value_type& _value);

        ~Tensor();


        const coords_t&  size() const;
        size_t size(size_t _dimension) const;
        size_t nDimensions() const;
        size_t nElements() const;

        void resize(const coords_t&  _dimensions);
        void resize(const coords_t&  _dimensions, const value_type& _value);

        void resizeAs(const Tensor& _tensor);
        void resizeAs(const Tensor& _tensor, const value_type& _value);

        void fill(value_type _value);
        void randomize(value_type _min = 0.0, value_type _max = 1.0);

        Tensor getTranspose() const;
        size_t getIndex(const coords_t&  _indices) const;

        value_type length() const;
        value_type length2() const;

        value_type max() const;
        coords_t  argmax() const;

        void print(bool _return = true) const;

        value_type& operator[](size_t  _index);
        const value_type& operator[](size_t  _index) const;

        value_type& operator()(coords_t  _indices);
        value_type& operator()(size_t i0);
        value_type& operator()(size_t i0, size_t i1);
        value_type& operator()(size_t i0, size_t i1, size_t i2);

        const value_type& operator()(coords_t  _indices) const;
        const value_type& operator()(size_t i0) const;
        const value_type& operator()(size_t i0, size_t i1) const;
        const value_type& operator()(size_t i0, size_t i1, size_t i2) const;

        void operator+=(const Tensor& _tensor);
        void operator-=(const Tensor& _tensor);

        void addOuterProduct(const Tensor& a, const Tensor& b);

    private:
        coords_t  dimensions;
        coords_t  strides;
        std::vector<value_type> values;
};

void convolve(Tensor& result, const Tensor& kernel, const Tensor& src);

void dot(value_type& result, const Tensor& a, const Tensor& b);
void outerProduct(Tensor& result, const Tensor& a, const Tensor& b);

void mulmm(Tensor& result, const Tensor& a, const Tensor& b);
void mulmv(Tensor& result, const Tensor& a, const Tensor& b);

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);

Tensor operator*(const double& a, const Tensor& b);

std::ostream& operator<<(std::ostream& os, const Tensor& t);

Tensor Vector(std::initializer_list<value_type> _data);
Tensor Matrix(std::initializer_list<std::initializer_list<value_type>> _data);
