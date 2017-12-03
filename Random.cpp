#include "Random.h"

#include <ctime>
#include <cstdlib>

#include <iostream>

long int Random::seed = 0;

int Random::nextInt(int _min, int _max)
{
    return _min + (rand() % (_max - _min));
}

float Random::nextFloat(float _min, float _max)
{
    return _min +  (_max-_min) * (float)rand() / RAND_MAX;
}

double Random::nextDouble(double _min, double _max)
{
    return _min + (_max - _min) * (double)rand() / RAND_MAX;
}

void Random::setSeed()
{
    setSeed( static_cast<long int>(time(nullptr)) );
}

void Random::setSeed(long int _seed)
{
    seed = _seed;
    srand(seed);
}

long int Random::getSeed()
{
    return seed;
}
