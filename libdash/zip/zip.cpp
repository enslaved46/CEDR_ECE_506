#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "dma.h"
#include "zip.h"

#if !(DASH_PLATFORM == DASH_ZCU102_2FFT_2MMULT_1ZIP_1CONV2D_HWSCHEDULER || DASH_PLATFORM == DASH_ZEDBOARD || DASH_PLATFORM == DASH_AUP_ZU3)
  #error "ZIP is not supported on this platform!"
#else

#define SEC2NANOSEC 1000000000

static volatile unsigned int* zip_control_base_addr[NUM_ZIPS];
static volatile unsigned int* dma_control_base_addr[NUM_ZIPS];
static volatile unsigned int* udmabuf_base_addr;
static uint64_t               udmabuf_phys_addr;
#if defined(ZIP_RESET_BASE_ADDRS)
static volatile unsigned int* zip_reset_base_addr[NUM_ZIPS];
#endif

void __attribute__((constructor)) setup_zip(void) {
  LOG("[zip] Running ZIP constructor\n");

  for(uint8_t i = 0; i < NUM_ZIPS; i++){
    LOG("[zip] Initializing ZIP DMA (0x%x)\n", ZIP_DMA_CTRL_BASE_ADDRS[i]);
    dma_control_base_addr[i] = init_dma(ZIP_DMA_CTRL_BASE_ADDRS[i]);
    #if defined(ZIP_RESET_BASE_ADDRS)
    zip_reset_base_addr[i] = init_acc_reset(ZIP_RESET_BASE_ADDRS[i]);
    reset_acc_and_dma(zip_reset_base_addr[i]);
    #else
    reset_dma(dma_control_base_addr[i]);
    #endif

    LOG("[zip] Initializing ZIP\n");
    zip_control_base_addr[i] = init_zip(ZIP_CONTROL_BASE_ADDRS[i]);
  }
  
  LOG("[zip] Initializing udmabuf\n");
  init_udmabuf(ZIP_UDMABUF_NUM, ZIP_UDMABUF_SIZE, &udmabuf_base_addr, &udmabuf_phys_addr);
  LOG("[zip] udmabuf base address is 0x%p\n", udmabuf_base_addr);

  LOG("[zip] ZIP constructor complete!\n");
}

void __attribute__((destructor)) teardown_zip(void) {
  LOG("[zip] Running ZIP destructor\n");
  close_udmabuf(udmabuf_base_addr, ZIP_UDMABUF_SIZE);

  for (uint8_t i = 0; i < NUM_ZIPS; i++){
    close_zip(zip_control_base_addr[i]);
    close_dma(dma_control_base_addr[i]);
    #if defined(ZIP_RESET_BASE_ADDRS)
    close_acc_reset(zip_reset_base_addr[i]);
    #endif
  }
  LOG("[zip] ZIP destructor complete!\n");
}

void zip_accel(zip_re_type* input0, zip_re_type* input1, zip_re_type* output, size_t size, int op, uint8_t resource_idx) {
  LOG("[zip-%u] Running zip for vector size %ld on the FPGA\n", resource_idx, size);
  //struct timespec start_accel {};
  //struct timespec end_accel {};

  volatile unsigned int *dma_control_base = dma_control_base_addr[resource_idx];
  volatile unsigned int *zip_control_base = zip_control_base_addr[resource_idx];

  volatile unsigned int *udmabuf_base = udmabuf_base_addr + (resource_idx * (UDMABUF_PARTITION_SIZE / sizeof(unsigned int)));
  uint64_t udmabuf_phys = udmabuf_phys_addr + (resource_idx * UDMABUF_PARTITION_SIZE);

  LOG("[zip-%u] zip control reg is %p\n", resource_idx, zip_control_base);
  LOG("[zip-%u] dma control reg is %p\n", resource_idx, dma_control_base);
  LOG("[zip-%u] udmabuf base address is 0x%x\n", resource_idx, (unsigned int) *udmabuf_base);
  LOG("[zip-%u] udmabuf phys address is 0x%lx\n", resource_idx, udmabuf_phys);

  size_t i;
  #if defined(ZIP_RESET_BASE_ADDRS)
  volatile unsigned int *zip_reset_base = zip_reset_base_addr[resource_idx];
  LOG("[zip-%u] zip gpio reset base address is %p\n", resource_idx, zip_reset_base);
  LOG("[zip-%u] Using GPIO to reset both ZIP and ZIP DMA IP cores...\n", resource_idx);
  reset_acc_and_dma(zip_reset_base);
  #else
  LOG("[zip-%u] Resetting DMA engine\n", resource_idx);
  reset_dma(dma_control_base);
  #endif
  LOG("[zip-%u] Resetting DMA engine done!\n", resource_idx);

  switch(op){
    case 0: // ZIP_ADD
      LOG("[zip-%u] Configuring ZIP as ADD\n", resource_idx);
      break;
    case 1: // ZIP_SUB
      LOG("[zip-%u] Configuring ZIP as SUB\n", resource_idx);
      break;
    case 2: // ZIP_MULT
      LOG("[zip-%u] Configuring ZIP as MULT\n", resource_idx);
      break;
    case 3: // ZIP_DIV
      LOG("[zip-%u] Configuring ZIP as DIV\n", resource_idx);
      break;
    default:
      LOG("[zip-%u] ZIP operation not supported!\n", resource_idx);
      break;
  }
  config_zip_op(zip_control_base, op);

  LOG("[zip-%u] Configuring ZIP with size %ld\n", resource_idx, size);
  config_zip_size(zip_control_base, size);

  LOG("[zip-%u] Copying input0 buffer to udmabuf (udmabuf_base: %p, input0: %p)\n", resource_idx, udmabuf_base, input0);
  memcpy((float* ) udmabuf_base, input0, 2*size * sizeof(float));
  LOG("[zip-%u] Copying input0 buffer to udmabuf (udmabuf_base: %p, input0: %p)\n", resource_idx, udmabuf_base+size, input1);
  memcpy((float* ) udmabuf_base+(2*size), input1, 2*size * sizeof(float));

  LOG("[zip-%u] Calling setup_rx for dma\n", resource_idx);
  setup_rx(dma_control_base, udmabuf_phys + (4 * size * sizeof(float)), 2*size * sizeof(float));

  //clock_gettime(CLOCK_MONOTONIC_RAW, &start_accel);

  LOG("[zip-%u] Calling setup_tx for dma\n", resource_idx);
  setup_tx(dma_control_base, udmabuf_phys, 4 * size * sizeof(float));
  LOG("[zip-%u] Writing to control register\n", resource_idx);
  zip_write_reg(zip_control_base_addr[resource_idx], ZIP_AP_CTRL, ZIP_AP_START);
  
  LOG("[zip-%u] Waiting for RX to complete for dma0\n", resource_idx);
  dma_wait_for_rx_complete(dma_control_base);
  
  //clock_gettime(CLOCK_MONOTONIC_RAW, &end_accel);
  
//  LOG("[zip-%u] ZIP accelerator execution time (ns): %lf\n", resource_idx,
//         ((double)end_accel.tv_sec * SEC2NANOSEC + (double)end_accel.tv_nsec) - ((double)start_accel.tv_sec * SEC2NANOSEC + (double)start_accel.tv_nsec));

  LOG("[zip-%u] Memcpy output back\n", resource_idx);
  memcpy(output, &(((float*)udmabuf_base)[4 * size]), 2 * size * sizeof(float));
 
  LOG("[zip-%u] Finished ZIP on the FPGA\n", resource_idx);
}

extern "C" void DASH_ZIP_flt_zip(dash_cmplx_flt_type** x, dash_cmplx_flt_type** y, dash_cmplx_flt_type** z, size_t* size, int* op, uint8_t resource_idx){
  if(sizeof(dash_cmplx_flt_type) == sizeof(zip_cmplx_type)){
    int i=0;
    while( i+2048 < *size ){
      zip_accel((dash_re_flt_type *) &((*x)[i]), (dash_re_flt_type *) &((*y)[i]), (dash_re_flt_type *) &((*z)[i]), 2048, *op, resource_idx);
      i = i + 2048;
    }
    int new_size = *size - i;
    zip_accel((dash_re_flt_type *) &((*x)[i]), (dash_re_flt_type *) &((*y)[i]), (dash_re_flt_type *) &((*z)[i]), new_size, *op, resource_idx);
  }
  else{
    zip_cmplx_type* inp0 = (zip_cmplx_type*) malloc((*size) * sizeof(zip_cmplx_type));
    zip_cmplx_type* inp1 = (zip_cmplx_type*) malloc((*size) * sizeof(zip_cmplx_type));
    zip_cmplx_type* out = (zip_cmplx_type*) malloc((*size) * sizeof(zip_cmplx_type));

    for (size_t i = 0; i < (*size); i++) {
      inp0[i].re = (zip_re_type) ((*x)[i]).re;
      inp0[i].im = (zip_re_type) ((*x)[i]).im;
      inp1[i].re = (zip_re_type) ((*y)[i]).re;
      inp1[i].im = (zip_re_type) ((*y)[i]).im;
    }

    zip_accel((dash_re_flt_type *) inp0, (dash_re_flt_type *) inp1, (dash_re_flt_type *) out, (*size), *op, resource_idx);

    for (size_t i = 0; i < (*size); i++) {
      (*z)[i].re = (dash_re_flt_type) (out[i]).re;
      (*z)[i].im = (dash_re_flt_type) (out[i]).im;
    }

    free(inp0);
    free(inp1);
    free(out);
  }
}

#if defined(__ZIP_ENABLE_MAIN)
uint32_t _check_zip_result_(zip_re_type * a, zip_re_type *b, zip_re_type *zip_actual, zip_re_type *zip_expected, size_t size) {
  int error_count = 0;
  double diff;
  float c, d;

  LOG("[zip] Checking actual versus expected output\n");
  for (size_t i = 0; i < 2*size; i++) {
    c = zip_expected[i];
    d = (zip_re_type) zip_actual[i];

    diff = std::abs(std::abs(c - d) / c * 100);

    if (diff > 0.01) {
      LOG("[zip] ERROR %ld - In0 = %lf, In1 = %lf, Expected = %lf, Hardware ZIP = %lf, diff = %lf, size = %ld\n", i, a[i], b[i], c, d, diff, size);
      error_count++;
    }
  }

  if (error_count == 0) {
    LOG("[zip] ZIP Passed!\n");
    return 0;
  } else {
    fprintf(stderr, "[zip] ZIP Failed!\n");
    return 1;
  }
}

void _generate_zip_test_values_(zip_re_type *zip_input0, zip_re_type *zip_input1, zip_re_type *zip_expected, size_t h_len, int op) {
  for(int i = 0; i < 2*h_len; i++){
    switch(op){
      case 0: // ZIP_ADD
        zip_expected[i] = (zip_re_type) (zip_input0[i] + zip_input1[i]);
        break;
      case 1: // ZIP_SUB
        zip_expected[i] = (zip_re_type) (zip_input0[i] - zip_input1[i]);
        break;
      case 2: // ZIP_MULT
        zip_expected[i] = (zip_re_type) (zip_input0[i]*zip_input1[i] - zip_input0[i+1]*zip_input1[i+1]);
        zip_expected[i+1] = (zip_re_type) (zip_input0[i+1]*zip_input1[i] + zip_input0[i]*zip_input1[i+1]);
        i++;
        break;
      case 3: // ZIP_DIV
        zip_expected[i] = (zip_re_type) (zip_input0[i]*zip_input1[i] + zip_input0[i+1]*zip_input1[i+1]) / (zip_input1[i]*zip_input1[i] + zip_input1[i+1]*zip_input1[i+1]);
        zip_expected[i+1] = (zip_re_type) (-zip_input0[i]*zip_input1[i+1] + zip_input0[i+1]*zip_input1[i]) / (zip_input1[i]*zip_input1[i] + zip_input1[i+1]*zip_input1[i+1]);
        break;
    }
  }
}

int main() {
  zip_cmplx_type *A;
  zip_cmplx_type *B;
  zip_cmplx_type *C;
  zip_cmplx_type *C_expected;
  int op;
  int error_counter=0;
  
  size_t test_len = 2060;//1024;

  LOG("[zip] Allocating space for zip tests - A\n");
  A = (zip_cmplx_type*) calloc(test_len, sizeof(zip_cmplx_type));
  LOG("[zip] Allocating space for zip tests - B\n");
  B = (zip_cmplx_type*) calloc(test_len, sizeof(zip_cmplx_type));
  LOG("[zip] Allocating space for zip tests - C\n");
  C = (zip_cmplx_type*) calloc(test_len, sizeof(zip_cmplx_type));
  LOG("[zip] Allocating space for zip tests - C_expect\n");
  C_expected = (zip_cmplx_type*) calloc(test_len, sizeof(zip_cmplx_type));
  LOG("[zip] Completed allocating space for zip tests\n");

  int base = 4;
  LOG("[zip] Starting trial sweep\n");
  for (int j=0; j<10; j++){
    LOG("[zip] Running trial %u\n", j);
  for (uint8_t zip_num = 0; zip_num < NUM_ZIPS; zip_num++) {
    LOG("[zip-%u] Setting values for trial %u\n", zip_num, j);
  for (int i = 0; i < test_len; i++){
    A[i].re = (zip_re_type) base*(i+1)/100;
    A[i].im = (zip_re_type) base*(2*i+1)/100;
    B[i].re = (zip_re_type) (base*2)*(i+1)/100;
    B[i].im = (zip_re_type) (base*2)*(2*i+1)/100;
  }
    LOG("[zip-%u] Beginning tests on accelerator %u\n", zip_num, zip_num);
    // Test ZIP add
    op = 0;
    LOG("[zip-%u] Testing ZIP Add...\n", zip_num);
    _generate_zip_test_values_((dash_re_flt_type*) A, (dash_re_flt_type*) B, (dash_re_flt_type*) C_expected, test_len, 0);
    DASH_ZIP_flt_zip((dash_cmplx_flt_type**) &A, (dash_cmplx_flt_type**) &B, (dash_cmplx_flt_type**) &C, &test_len, &op, zip_num);
    error_counter+=_check_zip_result_((dash_re_flt_type*) A, (dash_re_flt_type*) B, (dash_re_flt_type*) C, (dash_re_flt_type*) C_expected, test_len);

     // Test ZIP sub
    op = 1;
    LOG("[zip-%u] Testing ZIP Sub...\n", zip_num);
    _generate_zip_test_values_((dash_re_flt_type*) A, (dash_re_flt_type*) B, (dash_re_flt_type*) C_expected, test_len, 1);
    DASH_ZIP_flt_zip((dash_cmplx_flt_type**) &A, (dash_cmplx_flt_type**) &B, (dash_cmplx_flt_type**) &C, &test_len, &op, zip_num);
    error_counter+=_check_zip_result_((dash_re_flt_type*) A, (dash_re_flt_type*) B, (dash_re_flt_type*) C, (dash_re_flt_type*) C_expected, test_len);

    // Test ZIP mult
    op = 2;
    LOG("[zip-%u] Testing ZIP Mult...\n", zip_num);
    _generate_zip_test_values_((dash_re_flt_type*) A, (dash_re_flt_type*) B, (dash_re_flt_type*) C_expected, test_len, 2);
    DASH_ZIP_flt_zip((dash_cmplx_flt_type**) &A, (dash_cmplx_flt_type**) &B, (dash_cmplx_flt_type**) &C, &test_len, &op, zip_num);
    error_counter+=_check_zip_result_((dash_re_flt_type*) A, (dash_re_flt_type*) B, (dash_re_flt_type*) C, (dash_re_flt_type*) C_expected, test_len);
/*
    // Test ZIP div
    op = 3;
    LOG("[zip-%u] Testing ZIP Div...\n", zip_num);
    _generate_zip_test_values_(A, B, C_expected, test_len, 3);
    DASH_ZIP_flt_zip(&A, &B, &C, &test_len, &op, zip_num);
    error_counter+=_check_zip_result_((dash_cmplx_flt_type**) A, (dash_cmplx_flt_type**) B, (dash_cmplx_flt_type**) C, C_expected, test_len);
*/  }
  base = base*2;
  }
  LOG("[zip] There are %d errors!\n", error_counter);
  free(A); free(B); free(C); free(C_expected);
}
#endif

#endif
