__kernel void mul(__global float* _tensor, float _scalar, int _stride)
{
    const int index = get_global_id(0)*_stride;

    for (int k = 0; k < _stride; ++k)
        _tensor[index + k] = _scalar * _tensor[index + k];
}
