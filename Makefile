all: build run

build:
	@mpicc -mavx2 -Ofast -fopenmp matmul.c -lm -o matmul

run:
	@OMP_NUM_THREADS=4 mpiexec -np 4 ./matmul 2048 2048 2048 128 128 128 8 8 4 4 1

clean:
	rm matmul
