#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

// Macros para facilitar escribir el acceso a los elementos de las matrices
#define A(i,j) A[(i)+m*(j)]
#define B(i,j) B[(i)+k*(j)]
#define C(i,j) C[(i)+m*(j)]
#define Cseq(i,j) Cseq[(i)+m*(j)]

#define Atemp(i,j) Atemp[(i)+m*(j)]
#define Btemp(i,j) Btemp[(i)+k*(j)]

#define Alocal(i,j) Alocal[(i)+mb*(j)]
#define Blocal(i,j) Blocal[(i)+kb*(j)]
#define Clocal(i,j) Clocal[(i)+mb*(j)]

#define Agroup(i,j) Agroup[(i)+mb*(j)]
#define Bgroup(i,j) Bgroup[(i)+kb*(j)]

#include "packings.c"

void printmat( double* mat, int rowsize, int colsize );
void microkernel( const int mr, const int nr, const int kc, double *Cc, const int m, double *Ac, double *Bc );

#include "microkernel_AVX_4x8_fp64.c"

// Como asumimos que cada particion de Cij, Aij y Bij se asigna al proceso Pij de la malla, será esta malla la que determine el tamaño de bloque
int main( int argc, char** argv ) {
    int rank, size;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    unsigned int seed = rank; // semilla para rand_r

    double elapsed = 0; // variable para calcular el tiempo

    // Recibir argumentos para la matriz
    if ( argc < 11 ) {
        if ( rank == 0 ) printf( "Usage: %s <m> <n> <k> <mc> <nc> <kc> <mr> <nr> <grid_nrows> <grid_ncols> <seq?>\n", argv[0] );
        MPI_Finalize();
        return -1;
    }

    int m,n,k, mb,nb,kb, mc,nc,kc, mr,nr, nrows,ncols, seq = 0;
    sscanf( argv[1], "%d", &m );
    sscanf( argv[2], "%d", &n );
    sscanf( argv[3], "%d", &k );
    sscanf( argv[4], "%d", &mc );
    sscanf( argv[5], "%d", &nc );
    sscanf( argv[6], "%d", &kc );
    sscanf( argv[7], "%d", &mr );
    sscanf( argv[8], "%d", &nr );
    sscanf( argv[9], "%d", &nrows );
    sscanf( argv[10], "%d", &ncols );
    if ( argc > 11 ) { sscanf( argv[11], "%d", &seq ); } // Calculo secuencial opcional

    mb = m / nrows;
    nb = n / ncols;
    kb = k / ncols; // como la malla va a ser cuadrada, da igual divirlo entre ncols o nrows
    // int blockcount = n / blocksize;
    int N = nrows; // como la malla va a ser cuadrada, quien determina N en una matriz cuadrada es nrows

    // Comprobaciones de que no hay ningun parametro mal
    if ( (k % (int)sqrt(size) != 0) && (n % (int)sqrt(size) != 0) && (m % (int)sqrt(size) != 0) ) { 
        if ( rank == 0 ) printf( "ERROR: m, n y k tienen que ser dividibles por sqrt(%d)\n", size);
        MPI_Finalize();
        return -1;
    }
    if ( nrows * ncols > size ) {
        if ( rank == 0 ) printf( "ERROR: El numero de procesos (%d) debe ser sqrt( nrows*ncols(%d) )\n",size,nrows*ncols);
        MPI_Finalize();
        return -1;
    }

    double* A; double* B; double* C; double* Cseq;
    A = malloc( m*k*sizeof(double) );
    B = malloc( k*n*sizeof(double) );
    C = calloc( m*n, sizeof(double) );
    if ( seq ) Cseq = calloc( m*n, sizeof(double) );

    // Generacion de matrices A y B;
    if ( rank == 0 ) {
        for ( int i = 0; i < m; i++ ) {
            for ( int j = 0; j < k; j++ ) {
                A(i,j) = ( (double) rand_r(&seed) ) / RAND_MAX;
            }
        }
        for ( int i = 0; i < k; i++ ) {
            for ( int j = 0; j < n; j++ ) {
                B(i,j) = ( (double) rand_r(&seed) ) / RAND_MAX;
            }
        }
    }

#ifdef PRINTMAT
    if ( rank == 0 ) {
        printf(" MATRIZ A: \n");
        printmat(A, m, k);
        printf(" MATRIZ B: \n");
        printmat(B, n, k);
    }
#endif

    // Multiplicacion secuencial opcional
    if ( seq && rank == 0 ) {
        elapsed = MPI_Wtime();
        for (int i = 0; i < m; i++) {
            for (int l = 0; l < k; l++) {
                for (int j = 0; j < n; j++) {
                    Cseq(i,j) = Cseq(i,j) + A(i,l) * B(l,j);
                }
            }
        }
        elapsed = MPI_Wtime() - elapsed;
        printf("Tiempo de ejecucion secuencial: %.4f segundos\n", elapsed);
    }
    
#ifdef PRINTMAT
    MPI_Barrier( MPI_COMM_WORLD );
    if ( rank == 0 ) {
        printf("MATRIZ Cseq: \n");
        printmat(Cseq, n, m);
    }
#endif

    // Creacion de malla cartesiana de procesos
    const int ndims = 2;
    MPI_Comm comm;
    int dims[ndims]; dims[0] = nrows; dims[1] = ncols;
    int wrap[ndims]; wrap[0] = 0; wrap[1] = 0;
    int reorder = 0;
    int coords[ndims];
    MPI_Cart_create( MPI_COMM_WORLD, ndims, dims, wrap, reorder, &comm );
    MPI_Cart_coords( comm, rank, ndims, coords );

    // Creacion de comunicadores de filas y columnas
    MPI_Comm commrow, commcol;
    int belongs[ndims];
    belongs[0] = 0; belongs[1] = 1;
    MPI_Cart_sub( comm, belongs, &commrow );
    belongs[0] = 1; belongs[1] = 0;
    MPI_Cart_sub( comm, belongs, &commcol );

    // Obtener rank en comunicadores de filas y columnas
    int rankrow, rankcol;
    MPI_Comm_rank ( commrow, &rankrow );
    MPI_Comm_rank ( commcol, &rankcol );

    double* Atemp, *Alocal;
    double* Btemp, *Blocal;
    double* Clocal;
    double* Agroup, *Bgroup;

    Atemp = malloc( kb*m * sizeof(double) );
    Btemp = malloc( nb*k * sizeof(double) );
    Alocal = malloc( mb*kb * sizeof(double) );
    Blocal = malloc( kb*nb * sizeof(double) );
    Clocal = calloc( mb*nb, sizeof(double) );
    Agroup = malloc( mb*kb * sizeof(double) );
    Bgroup = malloc( kb*nb * sizeof(double) );

    MPI_Datatype MPI_BLOQUE_A_1, MPI_BLOQUE_A_2;
    MPI_Datatype MPI_BLOQUE_B_1, MPI_BLOQUE_B_2;
    MPI_Datatype MPI_BLOQUE_C_1, MPI_BLOQUE_C_2;

    MPI_Type_vector( kb, mb, m, MPI_DOUBLE, &MPI_BLOQUE_A_1 );
    MPI_Type_commit( &MPI_BLOQUE_A_1 );
    MPI_Type_create_resized( MPI_BLOQUE_A_1, 0, mb*sizeof(double), &MPI_BLOQUE_A_2);
    MPI_Type_commit( &MPI_BLOQUE_A_2 );

    MPI_Type_vector( nb, kb, k, MPI_DOUBLE, &MPI_BLOQUE_B_1 );
    MPI_Type_commit( &MPI_BLOQUE_B_1 );
    MPI_Type_create_resized( MPI_BLOQUE_B_1, 0, kb*sizeof(double), &MPI_BLOQUE_B_2);
    MPI_Type_commit( &MPI_BLOQUE_B_2 );

    MPI_Type_vector( nb, mb, m, MPI_DOUBLE, &MPI_BLOQUE_C_1 );
    MPI_Type_commit( &MPI_BLOQUE_C_1 );
    MPI_Type_create_resized( MPI_BLOQUE_C_1, 0, mb*sizeof(double), &MPI_BLOQUE_C_2);
    MPI_Type_commit( &MPI_BLOQUE_C_2 );


    elapsed = MPI_Wtime();

    // Distribucion inicial de datos de A, B y C
    if ( rankcol == 0 ) {
        MPI_Scatter( A, kb*m, MPI_DOUBLE, Atemp, kb*m, MPI_DOUBLE, 0, commrow );
        MPI_Scatter( B, nb*k, MPI_DOUBLE, Btemp, nb*k, MPI_DOUBLE, 0, commrow );
    }

    MPI_Scatter( Atemp , 1, MPI_BLOQUE_A_2, Alocal, mb*kb, MPI_DOUBLE, 0, commcol );
    MPI_Scatter( Btemp , 1, MPI_BLOQUE_B_2, Blocal, kb*nb, MPI_DOUBLE, 0, commcol );

#ifdef PRINTMAT
    MPI_Barrier( comm );
    for ( int i = 0; i < nrows; i++ ) {
        for ( int j = 0; j < ncols; j++ ) {
            MPI_Barrier( comm );
            if ( coords[0] == 0 && coords[0] == i && coords[1] == j ) {
                printf("PROC(%d,%d) MATRIZ Atemp: \n", coords[0], coords[1]);
                printmat(Atemp, kb, m);
            }
            if ( coords[0] == i && coords[1] == j ) {
                printf("PROC(%d,%d) MATRIZ Alocal: \n", coords[0], coords[1]);
                printmat(Alocal, kb, mb);
            }
        }
    }
#endif

#ifdef PRINTMAT
    MPI_Barrier( comm );
    for ( int i = 0; i < nrows; i++ ) {
        for ( int j = 0; j < ncols; j++ ) {
            MPI_Barrier( comm );
            if ( coords[0] == 0 && coords[0] == i && coords[1] == j ) {
                printf("PROC(%d,%d) MATRIZ Btemp: \n", coords[0], coords[1]);
                printmat(Btemp, nb, k);
            }
            if ( coords[0] == i && coords[1] == j ) {
                printf("PROC(%d,%d) MATRIZ Blocal: \n", coords[0], coords[1]);
                printmat(Blocal, nb, kb);
            }
        }
    }
#endif

    double* Ac, *Bc;
    Ac = malloc( mc * kc * sizeof(double) );
    Bc = malloc( kc * nc * sizeof(double) );

    // int m, nn, k;
    int jc,pc,ic;
    int bjj, bpp, app, aii;
    int j,p,i;
    int jr,ir;

    // Multiplicacion SUMMA
    for ( int bc = 0; bc < N; bc++ ) {
        if ( rankrow == bc ) memcpy( Agroup, Alocal, mb * kb * sizeof(double)); // copia datos en *group para poder compartirlos con Bcast
        if ( rankcol == bc ) memcpy( Bgroup, Blocal, kb * nb * sizeof(double)); // copia datos en *group para poder compartirlos con Bcast
        // Bcast datos A en fila
        MPI_Bcast( Agroup, mb*kb, MPI_DOUBLE, bc, commrow );
        // Bcast datos B en fila
        MPI_Bcast( Bgroup, kb*nb, MPI_DOUBLE, bc, commcol );

        // Multiplicacion de matrices local
        for (jc = 0; jc < nb; jc += nc) {
            for (pc = 0; pc < kb; pc += kc) {
                packB(kc,nc,nr,&Bgroup(pc,jc),kb,Bc);
                for (ic = 0; ic < mb; ic += mc) {
                    packA(mc,kc,mr,&Agroup(ic,pc),mb,Ac);
                    #pragma omp parallel for private(jr,ir)
                    for ( jr = 0; jr<nc; jr+=nr ) {
                        for ( ir = 0; ir<mc; ir+=mr ) {
                            // microkernel( mr, nr, kc, &Clocal(ic+ir,jc+jr), mb, &Ac[ir*kc], &Bc[kc*jr] );
                            microkernel_AVX_4x8_fp64( mr, nr, kc, &Clocal(ic+ir,jc+jr), mb, &Ac[ir*kc], &Bc[jr*kc]);
                        }
                    }
                }
            }
        }
    }

#ifdef PRINTMAT
    MPI_Barrier( comm );
    for ( int i = 0; i < nrows; i++ ) {
        for ( int j = 0; j < ncols; j++ ) {
            if ( coords[0] == i && coords[1] == j ) {
                printf("PROC(%d,%d) MATRIZ Clocal: \n", coords[0], coords[1]);
                printmat(Clocal, nb, mb);
            }
        }
    }
#endif

    // Recogida de datos
    double* Ctemp = malloc( nb * m * sizeof(double) );
    MPI_Gather( Clocal, mb*nb, MPI_DOUBLE, Ctemp, 1, MPI_BLOQUE_C_2, 0, commcol);
    if ( rankcol == 0 ) {
        MPI_Gather( Ctemp, nb*m, MPI_DOUBLE, C, nb*m, MPI_DOUBLE, 0, commrow);
    }

    elapsed = MPI_Wtime() - elapsed;
    if ( rank == 0 ) printf("Tiempo de ejecucion paralelo: %.4f segundos\n", elapsed);

#ifdef PRINTMAT
    MPI_Barrier( MPI_COMM_WORLD );
    if ( rank == 0 ) {
        printf("MATRIZ C: \n");
        printmat(C, n, m);
    }
#endif

    // Comprobacion de errores
    if ( seq && rank == 0 ) {
        double error = 0.0;
        for ( int i = 0; i < m*n; i++ ) {
            error += fabs(C[i]-Cseq[i]);
        }
        printf("Error = %.2e\n",error);
    }

    free( Ac );
    free( Bc );

    free( A );
    free( B );
    free( C );
    if ( seq ) free( Cseq );

    free( Atemp );
    free( Btemp );
    free( Ctemp );
    free( Alocal );
    free( Blocal );
    free( Clocal );
    free( Agroup );
    free( Bgroup );


    MPI_Type_free( &MPI_BLOQUE_A_1 );
    MPI_Type_free( &MPI_BLOQUE_A_2 );
    MPI_Type_free( &MPI_BLOQUE_B_1 );
    MPI_Type_free( &MPI_BLOQUE_B_2 );
    MPI_Type_free( &MPI_BLOQUE_C_1 );
    MPI_Type_free( &MPI_BLOQUE_C_2 );
    MPI_Comm_free( &commrow );
    MPI_Comm_free( &commcol );
    MPI_Comm_free( &comm );
    MPI_Finalize();
    return 0;
}

void microkernel( const int mr, const int nr, const int kc, double *Cc, const int m, double *Ac, double *Bc ) {
  double cr[mr*nr];
  int i= 0;
  int j = 0;
  int pr = 0;
  int ind = 0;

  // inicializar cr
  for ( i = 0; i < mr; i++ ) {
    for ( j = 0; j < nr; j++) {
      cr[ind++] = Cc[i+j*m];
    }
  }

  // calcular cr
  for ( pr = 0; pr < kc; pr++ ) {
    ind = 0;
    for ( i = 0; i < mr; i++ ) {
      for ( j = 0; j < nr; j++) {
         // Cc[i+j*m] += Ac[i+pr*mr] * Bc[j+pr*nr];
         cr[ind++] += Ac[i+pr*mr] * Bc[j+pr*nr];
      }
    }
  }

  // copiar cr a Cc
  ind = 0;
  for ( i = 0; i < mr; i++ ) {
    for ( j = 0; j < nr; j++) {
      Cc[i+j*m] = cr[ind++];
    }
  }
}

void printmat( double* mat, int rowsize, int colsize ) {
    for ( int i = 0; i < colsize; i++ ) {
        for ( int j = 0; j < rowsize; j++ ) {
            printf("\t%.2f",mat[j*colsize + i]);
        }
        printf("\n");
    }
    printf("\n");
}