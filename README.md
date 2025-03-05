# matmul

Aplicación de multiplicación de matrices de tamaño MxN

Para la compilación de este programa es necesario tener instalado GCC y OpenMPI, en concreto, para
utilizar las herramientas `mpicc` y `mpiexec`. También es necesario compilarlo en una plataforma que
soporte el uso de la extensión AVX-2 para la ejecución de las instrucciones SIMD utilizadas en la
optimización.

Una vez todo se tenga instalado solo es necesario hacer `make build` para la compilación y
`make run` para la ejecución.
