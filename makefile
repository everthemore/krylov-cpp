CC=g++
CFLAGS=-I eigen/ #~/Documents/physics/code/eigen/

Krylov: main.cpp
	$(CC) $(CFLAGS) -o Krylov -O3 -march=native -std=c++11 main.cpp
