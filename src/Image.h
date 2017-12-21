#pragma once

#include "Network.h"

Tensor loadImage(std::string _path);

void displayImages(std::vector<Tensor*> _images, std::vector<std::string> _names = {}, std::vector<unsigned> _pixelSizes = {});
void displayImage(const Tensor& _image, std::string _name = "Image", unsigned pixelSize = 1);

int ReverseInt (int i);
void LoadMNISTImages(rna::DataSet& _data); // 60000 examples
void LoadMNISTLabels(rna::DataSet& _data);
