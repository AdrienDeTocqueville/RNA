#pragma once

#include <map>
#include <string>
#include <sstream>

class FileReader
{
    public:
        FileReader(std::string _path);
        ~FileReader();

        template <typename T>
        T readAs(std::string _entry)
        {
            T val;
            operator[](_entry) >> val;

            return val;
        }

        std::istringstream& operator[](std::string _entry);

    private:
        std::string path;

        std::map<std::string, std::istringstream*> entries;
};
