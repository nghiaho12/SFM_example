#!/bin/sh

g++ -std=c++11 -O3 src/main.cpp -o main -I/usr/include/eigen3 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_calib3d -lopencv_imgcodecs -lgtsam -lboost_system -ltbb
