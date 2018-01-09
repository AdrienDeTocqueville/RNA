#include "Image.h"
#include "Utility/util.h"

#include <iostream>
#include <fstream>
#include <thread>

#include <SFML/Graphics.hpp>

Tensor loadImage(std::string _path)
{
    sf::Image im;
    if (!im.loadFromFile(_path))
    {
        std::cout << "Image not found" << std::endl;
        return Tensor{1, 0, 0};
    }

    size_t w = im.getSize().x;
    size_t h = im.getSize().y;

    std::cout << "Loading image of size: " << w << " x " << h << std::endl;

    Tensor t{3, w, h};

    for (unsigned i(0) ; i < w ; i++)
    {
        for (unsigned j(0) ; j < h ; j++)
        {
            t(0, i, j) = im.getPixel(i, j).r;
            t(1, i, j) = im.getPixel(i, j).g;
            t(2, i, j) = im.getPixel(i, j).b;

//            t(0, i, j) = t(0, i, j) / 255.0;
//            t(1, i, j) = t(1, i, j) / 255.0;
//            t(2, i, j) = t(2, i, j) / 255.0;

            t(0, i, j) = t(0, i, j) / 127.5 - 1.0;
            t(1, i, j) = t(1, i, j) / 127.5 - 1.0;
            t(2, i, j) = t(2, i, j) / 127.5 - 1.0;
        }
    }

    return t;
}

void displayImages(std::vector<Tensor*> _images, std::vector<std::string> _names, std::vector<unsigned> _pixelSizes)
{
    std::vector< std::thread > threads;

    for (unsigned i(0) ; i < _images.size() ; i++)
        threads.emplace_back(displayImage, *_images[i], (i < _names.size())?_names[i]: "Unnamed", (i < _pixelSizes.size())?_pixelSizes[i]: 1);

    for (unsigned i(0) ; i < _images.size() ; i++)
        threads[i].join();

}

// Note image is rotated relatively to console print
void displayImage(const Tensor& _image, std::string _name, unsigned pixelSize)
{
    unsigned w = 0;
    unsigned h = 0;

    if (_image.nDimensions() == 2)
    {
        w = _image.size(0);
        h = _image.size(1);
    }
    else
    {
        w = _image.size(1);
        h = _image.size(2);
    }

    // Create the main window
    sf::RenderWindow app(sf::VideoMode(w*pixelSize, h*pixelSize), _name);

    // Load a sprite to display
    sf::RectangleShape rect(sf::Vector2f(pixelSize, pixelSize));

    for (unsigned i(0) ; i < w ; i++)
    {
        for (unsigned j(0) ; j < h ; j++)
        {
            rect.setPosition(i*pixelSize, j*pixelSize);

            if (_image.nDimensions() == 2)
            {
                auto c = _image(i, j);

//                c = c * 255.0;
                c = (c +1.0)*127.5;

                c = clamp(0.0, c, 255.0);

                rect.setFillColor(sf::Color(c, c, c));
            }
            else if (_image.size(0) == 1)
            {
                auto c = _image(0, i, j);

//                c = c * 255.0;
                c = (c +1.0)*127.5;

                c = clamp(0.0, c, 255.0);

                rect.setFillColor(sf::Color(c, c, c));
            }
            else if (_image.size(0) == 3)
            {
                auto c1 = _image(0, i, j);
                auto c2 = _image(1, i, j);
                auto c3 = _image(2, i, j);

//                c1 *= 255.0;
//                c2 *= 255.0;
//                c3 *= 255.0;

                c1 = (c1 +1.0)*127.5;
                c2 = (c2 +1.0)*127.5;
                c3 = (c3 +1.0)*127.5;

                c1 = clamp(0.0, c1, 255.0);
                c2 = clamp(0.0, c2, 255.0);
                c3 = clamp(0.0, c3, 255.0);

                rect.setFillColor(sf::Color(c1, c2, c3));
            }

            app.draw(rect);
        }
    }

    app.display();

    while (app.isOpen())
    {
        sf::Event event;
        while (app.waitEvent(event))
        {
            if (event.type == sf::Event::Closed)
                app.close();
        }
    }
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

void LoadMNISTImages(rna::DataSet& _data, std::string _file)
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

            for (unsigned i(0) ; i < rows ; i++)
            {
                for (unsigned j(0) ; j < columns ; j++)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));

                    example.input(0, j, i) = (temp/255.0)*2.0 - 1.0;
                }
            }
        }
    }
    else
        std::cout << "Unable to open: " << _file << std::endl;
}

void LoadMNISTLabels(rna::DataSet& _data, std::string _file)
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
