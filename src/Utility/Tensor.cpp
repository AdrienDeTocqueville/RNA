#include "Tensor.h"

#include <iostream>
#include <cmath>

#include "Random.h"


std::ostream& operator<<(std::ostream& os, const coords_t& coords)
{
    if (!coords.size())
    {
        os << "()";
        return os;
    }

    os << "(" << coords[0];

    for (unsigned i(1) ; i < coords.size() ; i++)
        os << ", " << coords[i];

    os << ")";

    return os;
}

coords_t operator/(const coords_t& coords, const int& a)
{
    coords_t result(coords.size());

    for (unsigned i(0) ; i < coords.size() ; i++)
        result[i] = coords[i] / a;

    return result;
}

/// Tensor
Tensor Tensor::identityMatrix(size_t _dimension)
{
    Tensor id{_dimension, _dimension};

    for (unsigned i(0) ; i < _dimension ; i++)
        id(i, i) = 1.0;

    return id;
}

Tensor::Tensor():
    buffer(0)
{ }

Tensor::Tensor(Tensor&& _tensor):
    Tensor()
{
    swap(*this, _tensor);
}

Tensor::Tensor(const Tensor& _tensor):
    dimensions(_tensor.dimensions),
    strides(_tensor.strides),
    values(_tensor.values),
    buffer(0)
{ }

Tensor::Tensor(const coords_t& _dimensions, const value_type& _value):
    Tensor()
{
    resize(_dimensions, _value);
}

Tensor::Tensor(std::initializer_list<size_t> _dimensions, const value_type& _value):
    Tensor()
{
    resize(_dimensions, _value);
}

Tensor::~Tensor()
{
    releaseCL();
}

Tensor& Tensor::operator=(Tensor _tensor)
{
    swap(*this, _tensor);

    return *this;
}


void Tensor::openCL(const cl::Context& _context, cl_mem_flags _flags) const
{
    if (buffer)
        return;

    cl_int error = CL_SUCCESS;
    size_t byteSize = sizeof (value_type) * nElements();
    const void* host_ptr = (_flags&CL_MEM_USE_HOST_PTR || _flags&CL_MEM_COPY_HOST_PTR)? data(): nullptr;

    // TODO: Use clCreateImage2D for matrix ?
    buffer = clCreateBuffer(_context(), _flags, byteSize, const_cast<void*>(host_ptr), &error);

    if (error != CL_SUCCESS)
        std::cout << "Tensor::openCL() -> Unable to create buffer: " << error << std::endl;
}

void Tensor::releaseCL()
{
    clReleaseMemObject(buffer);
    buffer = 0;
}

const coords_t& Tensor::size() const
{
    return dimensions;
}

size_t Tensor::size(size_t _dimension) const
{
    #ifdef TENSOR_SAFE
        if (_dimension >= dimensions.size())
            std::cout << "Tensor::size() -> Out of range" << std::endl;
    #endif

    return dimensions[_dimension];
}

size_t Tensor::nDimensions() const
{
    return dimensions.size();
}

size_t Tensor::nElements() const
{
    return values.size();
}

void Tensor::resize(const coords_t& _dimensions, const value_type& _value)
{
    if (dimensions == _dimensions)
        return;

    releaseCL();

    if (!_dimensions.size())
    {
        dimensions.clear();
        strides.clear();
        values.clear();

        return;
    }

    dimensions = _dimensions;

    strides.resize(dimensions.size());
    strides.back() = 1;

    for (int i(dimensions.size()-2) ; i >= 0 ; i--)
        strides[i] = strides[i+1] * dimensions[i+1];

    values.resize(dimensions[0]*strides[0], _value);
}

void Tensor::resizeAs(const Tensor& _tensor, const value_type& _value)
{
    if (dimensions == _tensor.dimensions)
        return;

    releaseCL();

    dimensions = _tensor.dimensions;
    strides = _tensor.strides;
    values.resize(_tensor.values.size(), _value);
}

void Tensor::fill(value_type _value)
{
    for (value_type& v: values)
        v = _value;
}

void Tensor::round(unsigned _decimals)
{
    value_type factor = pow(value_type(10), _decimals);

    for (value_type& v: values)
        v = std::round(v*factor)/factor;
}

void Tensor::randomize(value_type _min, value_type _max)
{
    #ifdef TENSOR_SAFE
        if (_min >= _max)
            std::cout << "Tensor::randomize() -> Invalid range" << std::endl;
    #endif

    for (unsigned i(0) ; i < values.size() ; i++)
        values[i] = Random::nextFloat(_min, _max);
}

bool Tensor::isnan() const
{
    for (unsigned i(0) ; i < nElements() ; i++)
    {
        if (std::isnan(operator[](i)))
            return true;
    }

    return false;
}

Tensor Tensor::getTranspose() const
{
    #ifdef TENSOR_SAFE
        if (nDimensions() != 2)
            std::cout << "Unable to transpose this tensor" << std::endl;
    #endif

	Tensor transpose{dimensions[1], dimensions[0]};

    for (unsigned i(0) ; i < dimensions[0] ; i++)
        for (unsigned j(0) ; j < dimensions[1] ; j++)
			transpose(j, i) = operator()(i, j);

    return transpose;
}

size_t Tensor::getStride(size_t i) const
{
    #ifdef TENSOR_SAFE
        if (i >= strides.size())
            std::cout << "Tensor::getStride() -> Out of bounds" << std::endl;
    #endif

    return strides[i];
}

size_t Tensor::getIndex(const coords_t&  _indices) const
{
    #ifdef TENSOR_SAFE
        if (_indices.size() != dimensions.size())
            std::cout << "Tensor::getIndex() -> number of dimensions is invalid" << std::endl;

        for (unsigned i(0) ; i < dimensions.size() ; i++)
        {
            if (_indices[i] >= dimensions[i])
                std::cout << "Tensor::getIndex() -> index number " << i << " is out of range" << std::endl;
        }
    #endif

    size_t index = 0;
    for (unsigned i(0) ; i < _indices.size() ; i++)
        index += _indices[i] * strides[i];

    return index;
}

const cl_mem& Tensor::getBuffer() const
{
    #ifdef TENSOR_SAFE
        if (buffer == 0)
            std::cout << "Tensor::getBuffer() -> buffer is not yet created" << std::endl;
    #endif // TENSOR_SAFE

    return buffer;
}

void Tensor::setBuffer(cl_mem _buffer)
{
    buffer = _buffer;
}

void Tensor::readBuffer(cl::CommandQueue& _commandQueue, const cl_bool& _blockingRead) const
{
	cl_int error = clEnqueueReadBuffer(_commandQueue(), getBuffer(), _blockingRead, 0, nElements() * sizeof(value_type), const_cast<void*>((void*)data()), 0, nullptr, nullptr);

    #ifdef TENSOR_SAFE
        if (error != CL_SUCCESS)
            std::cout << "Tensor::readBuffer(): Unable to read buffer" << std::endl;
    #endif // TENSOR_SAFE
}

void Tensor::writeBuffer(cl::CommandQueue& _commandQueue, const cl_bool& _blockingWrite) const
{
	cl_int error = clEnqueueWriteBuffer(_commandQueue(), getBuffer(), _blockingWrite, 0, nElements() * sizeof(value_type), const_cast<void*>((void*)data()), 0, nullptr, nullptr);

    #ifdef TENSOR_SAFE
        if (error != CL_SUCCESS)
            std::cout << "Tensor::readBuffer(): Unable to read buffer" << std::endl;
    #endif // TENSOR_SAFE
}

Tensor::value_type* Tensor::data()
{
    return values.data();
}

const Tensor::value_type* Tensor::data() const
{
    return values.data();
}

Tensor::value_type Tensor::length() const
{
    return sqrt(length2());
}

Tensor::value_type Tensor::length2() const
{
    Tensor::value_type sum = 0.0;

    for (unsigned i(0) ; i < values.size() ; i++)
        sum += values[i]*values[i];

    return sum;
}

Tensor::value_type& Tensor::max()
{
    #ifdef TENSOR_SAFE
        if (!values.size())
            std::cout << "Tensor::max() -> empty tensor" << std::endl;
    #endif

    size_t index = 0;
	value_type maximum = values[0];

    for (unsigned i(1) ; i < values.size() ; i++)
    {
        if (values[i] > maximum)
        {
            index = i;
            maximum = values[i];
        }
    }

    return values[index];
}

const Tensor::value_type& Tensor::max() const
{
    #ifdef TENSOR_SAFE
        if (!values.size())
            std::cout << "Tensor::max() -> empty tensor" << std::endl;
    #endif

    size_t index = 0;
	value_type maximum = values[0];

    for (unsigned i(1) ; i < values.size() ; i++)
    {
        if (values[i] > maximum)
        {
            index = i;
            maximum = values[i];
        }
    }

    return values[index];
}

coords_t Tensor::argmax() const
{
    #ifdef TENSOR_SAFE
        if (!values.size())
            std::cout << "Tensor::argmax() -> empty tensor" << std::endl;
    #endif

    size_t index = 0;
	value_type maximum = values[0];

    for (unsigned i(1) ; i < values.size() ; i++)
    {
        if (values[i] > maximum)
        {
            index = i;
            maximum = values[i];
        }
    }

    coords_t arg(dimensions.size());

    for (unsigned i(0); i < dimensions.size(); i++)
    {
        arg[i] = index / strides[i];
        index -= arg[i] * strides[i];
    }

    return arg;
}

bool Tensor::operator==(const Tensor& _tensor)
{
    if (_tensor.size() != size())
        return false;

    for (unsigned i(0) ; i < nElements() ; i++)
    {
        if (operator[](i) != _tensor[i])
            return false;
    }

    return true;
}

bool Tensor::operator!=(const Tensor& _tensor)
{
    if (_tensor.size() != size())
        return true;

    for (unsigned i(0) ; i < nElements() ; i++)
    {
        if (operator[](i) != _tensor[i])
            return true;
    }

    return false;
}

Tensor::value_type& Tensor::operator[](size_t  _index)
{
    #ifdef TENSOR_SAFE
        if (_index >= values.size())
            std::cout << "Tensor::operator[]() -> index is out of range" << std::endl;
    #endif

    return values[_index];
}

const Tensor::value_type& Tensor::operator[](size_t _index) const
{
    #ifdef TENSOR_SAFE
        if (_index >= values.size())
            std::cout << "Tensor::operator[]() -> index is out of range" << std::endl;
    #endif

    return values[_index];
}

Tensor::value_type& Tensor::operator()(coords_t _indices)
{
    return values[getIndex(_indices)];
}

Tensor::value_type& Tensor::operator()(size_t i0)
{
    #ifdef TENSOR_SAFE
        return operator()( coords_t{i0} );
    #endif

    return values[i0];
}

Tensor::value_type& Tensor::operator()(size_t i0, size_t i1)
{
    #ifdef TENSOR_SAFE
        return operator()({i0, i1});
    #endif

    return values[i0*strides[0] + i1];
}

Tensor::value_type& Tensor::operator()(size_t i0, size_t i1, size_t i2)
{
    #ifdef TENSOR_SAFE
        return operator()({i0, i1, i2});
    #endif

    return values[i0*strides[0] + i1*strides[1] + i2];
}

const Tensor::value_type& Tensor::operator()(coords_t _indices) const
{
    return values[getIndex(_indices)];
}

const Tensor::value_type& Tensor::operator()(size_t i0) const
{
    #ifdef TENSOR_SAFE
        return operator()( coords_t{i0} );
    #endif

    return values[i0];
}

const Tensor::value_type& Tensor::operator()(size_t i0, size_t i1) const
{
    #ifdef TENSOR_SAFE
        return operator()({i0, i1});
    #endif

    return values[i0*strides[0] + i1];
}

const Tensor::value_type& Tensor::operator()(size_t i0, size_t i1, size_t i2) const
{
    #ifdef TENSOR_SAFE
        return operator()({i0, i1, i2});
    #endif

    return values[i0*strides[0] + i1*strides[1] +i2];
}

void Tensor::operator+=(const Tensor& _tensor)
{
    for (unsigned i(0) ; i < values.size() ; i++)
        values[i] += _tensor.values[i];
}

void Tensor::operator-=(const Tensor& _tensor)
{
    for (unsigned i(0) ; i < values.size() ; i++)
        values[i] -= _tensor.values[i];
}

void Tensor::addOuterProduct(const Tensor& a, const Tensor& b)
{
    #ifdef TENSOR_SAFE
        if (size(0) != a.size(0) || size(1) != b.size(0))
            std::cout << "Tensor::addOuterProduct() -> Result do not fit in tensor." << std::endl;
    #endif

    for (unsigned i(0) ; i < size(0) ; i++)
        for (unsigned j(0) ; j < size(1) ; j++)
            operator()(i, j) += a(i) * b(j);
}

Tensor::value_type dot(const Tensor& a, const Tensor& b)
{
    #ifdef TENSOR_SAFE
        if (a.size(0) != b.size(0))
            std::cout << "dot() -> sizes do not match" << std::endl;
    #endif

    Tensor::value_type result = 0.0;

    for (unsigned i(0) ; i < a.size(0) ; i++)
        result += a(i) * b(i);

    return result;
}

void convolve(Tensor& result, const Tensor& kernel, const Tensor& src)
{
    #ifdef TENSOR_SAFE
        if (src.size(0) != kernel.size(1))
            std::cout << "convolve() -> Kernel and source image don't have same the number of channels." << std::endl;
    #endif

    coords_t resSize{ kernel.size(0), src.size(1)-kernel.size(2)+1, src.size(2)-kernel.size(3)+1 };
    result.resize(resSize);

    unsigned mu = kernel.size(2)-1;
    unsigned mv = kernel.size(3)-1;

    for (unsigned k(0) ; k < result.size(0) ; k++)
    {
        for (unsigned i(0) ; i < result.size(1) ; i++)
        {
            for (unsigned j(0) ; j < result.size(2) ; j++)
            {
                result(k, i, j) = 0.0;

                for (unsigned c(0) ; c < kernel.size(1) ; c++)
                    for (unsigned u(0) ; u < kernel.size(2) ; u++)
                        for (unsigned v(0) ; v < kernel.size(3) ; v++)
                            result(k, i, j) += kernel({k, c, mu-u, mv-v}) * src(c, i+u, j+v);
            }
        }
    }
}

void outerProduct(Tensor& result, const Tensor& a, const Tensor& b)
{
    result.resize({a.size(0), b.size(0)});

    for (unsigned i(0) ; i < a.size(0) ; i++)
        for (unsigned j(0) ; j < b.size(0) ; j++)
            result(i, j) = a(i) * b(j);
}

void mulmm(Tensor& result, const Tensor& a, const Tensor& b)
{
    #ifdef TENSOR_SAFE
        if (a.size(1) != b.size(0))
            std::cout << "mulmm() -> sizes do not match" << std::endl;
    #endif

    result.resize({a.size(0), b.size(1)});

    for (unsigned i(0) ; i < a.size(0) ; i++)
    {
        for (unsigned j(0) ; j < b.size(1) ; j++)
        {
            auto& s = result(i, j); s = 0.0;

            for (unsigned k(0) ; k < a.size(1) ; k++)
                s += a(i, k) * b(k, j);
        }
    }
}

void mulmtm(Tensor& result, const Tensor& a, const Tensor& b)
{
    #ifdef TENSOR_SAFE
        if (a.size(0) != b.size(0))
            std::cout << "mulmtm() -> sizes do not match" << std::endl;
    #endif

    result.resize({a.size(1), b.size(1)});

    for (unsigned i(0) ; i < a.size(1) ; i++)
    {
        for (unsigned j(0) ; j < b.size(1) ; j++)
        {
            auto& s = result(i, j); s = 0.0;

            for (unsigned k(0) ; k < a.size(0) ; k++)
                s += a(k, i) * b(k, j);
        }
    }
}

void mulmmt(Tensor& result, const Tensor& a, const Tensor& b)
{
    #ifdef TENSOR_SAFE
        if (a.size(1) != b.size(1))
            std::cout << "mulmm() -> sizes do not match" << std::endl;
    #endif

    result.resize({a.size(0), b.size(0)});

    for (unsigned i(0) ; i < a.size(0) ; i++)
    {
        for (unsigned j(0) ; j < b.size(0) ; j++)
        {
            auto& s = result(i, j); s = 0.0;

            for (unsigned k(0) ; k < a.size(1) ; k++)
                s += a(i, k) * b(j, k);
        }
    }
}

void mulmv(Tensor& result, const Tensor& a, const Tensor& b)
{
    #ifdef TENSOR_SAFE
        if (a.size(1) != b.size(0))
            std::cout << "mulmv() -> sizes do not match" << std::endl;
    #endif

    result.resize({a.size(0)});

    for (unsigned i(0) ; i < a.size(0) ; i++)
    {
        result(i) = 0.0;

        for (unsigned k(0) ; k < a.size(1) ; k++)
            result(i) += a(i, k) * b(k);
    }
}

Tensor operator+(const Tensor& a, const Tensor& b)
{
    #ifdef TENSOR_SAFE
        if (a.size() != b.size())
            std::cout << "operator+() -> sizes do not match" << std::endl;
    #endif

    Tensor res;
    res.values.reserve(a.nElements());
    res.dimensions = a.dimensions;
    res.strides = a.strides;

    for (unsigned i(0) ; i < a.nElements() ; i++)
        res.values.push_back(a[i] + b[i]);

    return res;
}

Tensor operator-(const Tensor& a, const Tensor& b)
{
    #ifdef TENSOR_SAFE
        if (a.size() != b.size())
            std::cout << "operator-() -> sizes do not match" << std::endl;
    #endif

    Tensor res;
    res.values.reserve(a.nElements());
    res.dimensions = a.dimensions;
    res.strides = a.strides;

    for (unsigned i(0) ; i < a.nElements() ; i++)
        res.values.push_back(a[i] - b[i]);

    return res;
}

Tensor operator*(const Tensor::value_type& s, const Tensor& t)
{
    Tensor res;
    res.values.reserve(t.nElements());
    res.dimensions = t.dimensions;
    res.strides = t.strides;

    for (unsigned i(0) ; i < t.nElements() ; i++)
        res.values.push_back(s * t[i]);

    return res;
}

bool operator==(const Tensor& a, const Tensor& b)
{
    if (a.size() != b.size())
        return false;

    for (unsigned i(0) ; i < a.nElements() ; i++)
    {
        if (a[i] != b[i])
            return false;
    }

    return true;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t)
{
    #ifdef TENSOR_SAFE
        if (!t.nElements())
            std::cout << "operator<<() -> empty tensor" << std::endl;
    #endif

    if (t.nDimensions() == 1)
    {
        os << "(" << t(0);

        for (unsigned i(1) ; i < t.size(0) ; i++)
            os << " " << t(i);

        os << ")";
    }
    else if (t.nDimensions() == 2)
    {
        os << "[";

        for (unsigned j(0) ; j < t.size(1) ; j++)
            os << t(0, j) << " ";

        for (unsigned i(1) ; i < t.size(0) ; i++)
        {
            os << "\n";
            for (unsigned j(0) ; j < t.size(1) ; j++)
                os << " " << t(i, j);
        }

        os << "]";
    }
    else  if (t.nDimensions() == 3)
    {
        for (unsigned k(0); k < t.size(0); k++)
        {
            os << "[";

            for (unsigned j(0) ; j < t.size(2) ; j++)
                os << t(k, 0, j) << " ";

            for (unsigned i(1) ; i < t.size(1) ; i++)
            {
                os << "\n";
                for (unsigned j(0) ; j < t.size(2) ; j++)
                    os << " " << t(k, i, j);
            }

            os << "]" << "\n";
        }
    }
    else
        os << "Too many dimensions\n";

    return os;
}


Tensor Vector(std::initializer_list<Tensor::value_type> _data)
{
    Tensor t{_data.size()};

    size_t i(0);
    for (const Tensor::value_type& e: _data)
        t(i++) = e;

    return t;
}

Tensor Matrix(std::vector<Tensor>& _data)
{
    Tensor t{_data.size(), _data[0].size(0)};

    for (unsigned i(0) ; i < _data.size() ; i++)
    {
        for (unsigned j(0); j < _data[i].size(0); j++)
            t(i, j) = _data[i](j);
    }

    return t;
}

Tensor Matrix(std::initializer_list<Tensor> _data)
{
    const Tensor& d0(*_data.begin());
    Tensor t{_data.size(), d0.size(0)};

    size_t i(0);
    for (auto& r: _data)
    {
        for (unsigned j(0); j < t.size(1); j++)
            t(i, j) = r(j);

        i++;
    }

    return t;
}

Tensor Matrix(std::initializer_list<std::initializer_list<Tensor::value_type>> _data)
{
    const std::initializer_list<Tensor::value_type>& d0(*_data.begin());
    Tensor t{_data.size(), d0.size()};

    size_t i(0), j(0);
    for (auto& r: _data)
    {
        for (auto& c: r)
            t(i, j++) = c;

        i++;
        j = 0;
    }

    return t;
}
