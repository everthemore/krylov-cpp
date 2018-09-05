CC=g++
CFLAGS=-I eigen/ -fopenmp

Krylov: main.cpp
	$(CC) $(CFLAGS) -o Krylov -O3 -march=native -std=c++11 main.cpp
