#define cimg_use_jpeg                 // read .jpg files with libjpeg
#include "./CImg/CImg.h"
#include <cstdio>
#include <iostream>
using namespace cimg_library;

int main(int argc, char *argv[]) {
  if (argc == 1){
    std::cout << "File name missing. For instance, provide a jpg file" << std::endl;
    exit(-1);
  }

  CImg<unsigned char> img(argv[1]);                   // 8 bits per colour component
  printf("width %d, height %d, depth %d, channels %d\n", img.width(), img.height(), img.depth(), img.spectrum());

  int x = 18, y = 42;
  printf("pixel (%d, %d): R %d  G %d  B %d\n", x, y, img(x, y, 0, 0), img(x, y, 0, 1), img(x, y, 0, 2));

  img.display("my image");                                // opens a window; close it (or press Esc) to go on
}
