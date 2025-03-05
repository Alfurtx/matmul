// #include <intrin.h>
#include <immintrin.h>

#define MR 4
#define NR 8
inline __attribute__((always_inline)) void microkernel_AVX_4x8_fp64( const int mr, const int nr, const int kc, double *Clocal, const int mb, double *Ar, double *Br ) {

  __m256d C0, C1, C2, C3, C4, C5, C6, C7, A0, B0, B1;
  double *Aptr, *Bptr;
  int    baseB = 0, baseA = 0, Amr, Bnr;

  if( kc == 0 ) return;
  if( mr != MR && nr != NR ) return;

  C0 = _mm256_set1_pd ( 0 );
  C1 = _mm256_set1_pd ( 0 );
  C2 = _mm256_set1_pd ( 0 );
  C3 = _mm256_set1_pd ( 0 );
  C4 = _mm256_set1_pd ( 0 );
  C5 = _mm256_set1_pd ( 0 );
  C6 = _mm256_set1_pd ( 0 );
  C7 = _mm256_set1_pd ( 0 );

  Aptr = &Ar[0];
  Amr  = MR;
  Bptr = &Br[0];
  Bnr  = NR;

  for( int pr = 0; pr < kc; pr++ ) {
    A0 = _mm256_loadu_pd(&Aptr[baseA+0]);
    B0 = _mm256_loadu_pd(&Bptr[baseB+0]);
    B1 = _mm256_loadu_pd(&Bptr[baseB+4]);

    C0 += A0 * B0[0];
    C1 += A0 * B0[1];
    C2 += A0 * B0[2];
    C3 += A0 * B0[3];
    C4 += A0 * B1[0];
    C5 += A0 * B1[1];
    C6 += A0 * B1[2];
    C7 += A0 * B1[3];

    baseA = baseA + Amr;
    baseB = baseB + Bnr;
  }
  C0 += _mm256_loadu_pd(&Clocal(0,0));
  _mm256_storeu_pd ( &Clocal(0,0), C0 );
  C1 += _mm256_loadu_pd(&Clocal(0,1));
  _mm256_storeu_pd ( &Clocal(0,1), C1 );
  C2 += _mm256_loadu_pd(&Clocal(0,2));
  _mm256_storeu_pd ( &Clocal(0,2), C2 );
  C3 += _mm256_loadu_pd(&Clocal(0,3));
  _mm256_storeu_pd ( &Clocal(0,3), C3 );
  C4 += _mm256_loadu_pd(&Clocal(0,4));
  _mm256_storeu_pd ( &Clocal(0,4), C4 );
  C5 += _mm256_loadu_pd(&Clocal(0,5));
  _mm256_storeu_pd ( &Clocal(0,5), C5 );
  C6 += _mm256_loadu_pd(&Clocal(0,6));
  _mm256_storeu_pd ( &Clocal(0,6), C6 );
  C7 += _mm256_loadu_pd(&Clocal(0,7));
  _mm256_storeu_pd ( &Clocal(0,7), C7 );

}

