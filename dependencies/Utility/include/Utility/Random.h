#pragma once

#include <initializer_list>
#include <vector>

class Random
{
    public:
        static bool nextBool();

        // range is [_min, _max[
        static int nextInt(int _min = 0, int _max = 2);
        static float nextFloat(float _min = 0.0f, float _max = 1.0f);
        static double nextDouble(double _min = 0.0, double _max = 1.0);


        template <typename T>
        static T& element(std::initializer_list<T> _elements)
        {
            return *(_elements.begin() + Random::nextInt(0, _elements.size()));
        }

        template <typename T>
        static T& element(std::vector<T>& _elements)
        {
            return _elements[ Random::nextInt(0, _elements.size()) ];
        }

        template <typename T>
        static const T& element(const std::vector<T>& _elements)
        {
            return _elements[ Random::nextInt(0, _elements.size()) ];
        }


        static void init();
        static void setSeed(long int _seed);

        static long int getSeed();

    private:
        static long int seed;
};
