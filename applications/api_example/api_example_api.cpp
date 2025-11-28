#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <math.h>
#include "dash.h"

#include "../standalone_include/gsl_wrapper.h"

int main(void) {
  printf("[nk] Starting execution of FFTs\n");

  const size_t size = 256;
  dash_re_flt_type* input = (dash_re_flt_type*) malloc(2 * size * sizeof(dash_re_flt_type));
  dash_re_flt_type* output = (dash_re_flt_type*) malloc(2 * size * sizeof(dash_re_flt_type));
  dash_re_flt_type* output_verification = (dash_re_flt_type*) malloc(2 * size * sizeof(dash_re_flt_type));
  bool forwardTrans = true;

  for (int i = 0; i < 2 * size; i++) {
    input[i] = 1.0 * i;
  }

  gsl_fft_wrapper((dash_cmplx_flt_type*) input, (dash_cmplx_flt_type*) output_verification, size, forwardTrans);

  printf("[nk] Launching my kernel that was replaced from i.e. DASH_FFT\n");

  for (int i = 0; i < 10; i++) {
    printf("[nk] Launching FFT number %d\n", i);
    // gsl_fft_wrapper((dash_cmplx_flt_type*) input, (dash_cmplx_flt_type*) output, size, forwardTrans);
    DASH_FFT_flt((dash_cmplx_flt_type*) input, (dash_cmplx_flt_type*) output, size, forwardTrans);

    /* Do NOT Edit Below This Part! */
    for (int j = 0; j < 2 * size; j++) {
      if(fabs(output[j] - output_verification[j]) > 0.01){
        printf("[ERROR] Output of iteration %d is wrong!\n", i);
        break;
      }
    }
    /* Do NOT Edit Above This Part! */
  }

  free(input);
  free(output);
  free(output_verification);
  printf("[nk] Kernel execution for FFT is complete!\n");

  printf("\n\n");
  printf("[nk] Starting execution of ZIPs\n");
  
  // 256 compex vectors
  dash_cmplx_flt_type* A = (dash_cmplx_flt_type*) calloc(size, sizeof(dash_cmplx_flt_type));
  dash_cmplx_flt_type* B = (dash_cmplx_flt_type*) calloc(size, sizeof(dash_cmplx_flt_type));
  dash_cmplx_flt_type* C = (dash_cmplx_flt_type*) calloc(size, sizeof(dash_cmplx_flt_type));
  dash_cmplx_flt_type* C_verification = (dash_cmplx_flt_type*) calloc(size, sizeof(dash_cmplx_flt_type));
  for (int i = 0; i < size; i++) {
      A[i].re = (i);
      A[i].im = (i*2);
  }
  for (int i = 0; i < size; i++) {
      B[i].re = (i*3);
      B[i].im = (i*4);
  }

  for (int i = 0; i < size; i++) {
    C_verification[i].re = A[i].re * B[i].re - A[i].im * B[i].im;
    C_verification[i].im = A[i].re * B[i].im + A[i].im * B[i].re;
  }

  printf("[nk] Launching my kernel that was replaced from i.e. DASH_ZIP\n");
  for (int j = 0; j < 10; j++) {
    printf("[nk] Launching ZIP number %d for vector multiplication\n", j);
    // for (int i = 0; i < size; i++) {
    //   C[i].re = A[i].re * B[i].re - A[i].im * B[i].im;
    //   C[i].im = A[i].re * B[i].im + A[i].im * B[i].re;
    // }
    DASH_ZIP_flt(A, B, C, size, ZIP_MULT);
    /* Do NOT Edit Below This Part! */
    for (int i = 0; i < size; i++) {
      if( (fabs(C[i].re - C_verification[i].re) > 0.01) || (fabs(C[i].im - C_verification[i].im) > 0.01) ){
        printf("[ERROR] Output of iteration %d is wrong!\n", j);
        break;
      }
    }
    /* Do NOT Edit Above This Part! */
  }
  free(A);
  free(B);
  free(C);
  free(C_verification);
  printf("[nk] Kernel execution for ZIP is complete!\n");

  printf("[nk] Non-kernel thread execution is complete\n\n");
  return 0;
}
