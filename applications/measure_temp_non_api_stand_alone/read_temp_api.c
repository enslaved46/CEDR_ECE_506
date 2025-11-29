/* 
   Author  : Ashish Khadka
   Purpose : Poll FPD and LPD temperature until CEDR process is active in Kernel.
   Then write a temp log with time stamp when CEDR terminate.
   Program runtime args : argv[1] -> type of Temp conversion (K and C),
   argv[2] -> wrFile Prefix name (string name to idenify experment run types),
   argv[3] -> IDLE Board Temp meaurement type (IDLE or NOT_IDEAL)
*/
#include <stdio.h>
#include <stdlib.h>   // malloc
#include <unistd.h>   // usleep
#include <string.h>   // strcpy and cat
#include <time.h>     // time struct
#include <stdint.h>   // for types
#include <inttypes.h> // for print macro of unsigned long or u64
#include <stdbool.h>

//#define MAX_SAMPLES 1024*100
#define MAX_SAMPLES 1024*10000
#define DEBUG_PRINT 1
#define SLEEP_TIME  50           // in micro sec
/*
  raw  and offset are whole numbers
  scale is in fractions
*/

// global variable

// Define the states
typedef enum {
  INIT,
  MEASURE_LPD_TEMP,
  MEASURE_FPD_TEMP,
  SLEEP,
  CHECK_CEDR_PROCESS,
  RD_SAVED_SAMPLES,
  SAVE_SAMPLES_IN_FILE
} ece506TempRdState;

typedef enum {
  IDLE_TEMP_MEASUREMNT,                // IDLE Board temp measuremnt
  CEDR_RUN_TIME_MEASUREMENT       // CEDR Runtime temp measuremnt  
} tempMeasuremntType;
  
struct tempData{
  long     rawTempRd[MAX_SAMPLES];  // dont care abt the size
  uint64_t timeStamp[MAX_SAMPLES];
  int      numberOfSamplesRd;
  long     offset;
  double   scale;
};

// init struct and set max to 0
struct tempData lpdStructData = {.numberOfSamplesRd = 0, .offset = 0, .scale  = 0.0};
struct tempData fpdStructData = {.numberOfSamplesRd = 0, .offset = 0, .scale  = 0.0};

//struct tempData lpdStructData[MAX_SAMPLES];
//struct tempData fpdStructData[MAX_SAMPLES];


long read_long_frm_file(const char *path) {
  FILE *fp = fopen(path, "r");
  if (!fp) {
    perror("Failed to open file");
    exit(1);
  }

  long value;
  fscanf(fp, "%ld", &value);
  fclose(fp);
  return value;
}

double read_scaling_factor_frm_file(const char *path) {
  FILE *fp = fopen(path, "r");
  if (!fp) {
    perror("Failed to open file");
    exit(1);
  }

  double value;
  fscanf(fp, "%lf", &value);
  fclose(fp);
  return value;
}

void rdLPDconsts(struct tempData *lpdStructData){
  lpdStructData -> offset = read_long_frm_file("/sys/bus/iio/devices/iio:device0/in_temp7_offset");
  lpdStructData -> scale  = read_scaling_factor_frm_file("/sys/bus/iio/devices/iio:device0/in_temp7_scale");
}

void rdFPDconsts(struct tempData *fpdStructData){
  fpdStructData -> offset = read_long_frm_file("/sys/bus/iio/devices/iio:device0/in_temp8_offset");
  fpdStructData -> scale  = read_scaling_factor_frm_file("/sys/bus/iio/devices/iio:device0/in_temp8_scale");
}

void rdConsts(struct tempData *lpdStructData, struct tempData *fpdStructData){
  rdLPDconsts(lpdStructData); // pass ptr directly
  rdFPDconsts(fpdStructData);
}

void printFn(long raw, long offset, double scale, double temperature){
  printf("raw     = %ld\n", raw);
  printf("offset  = %ld\n", offset);
  printf("scale   = %f\n", scale);
  printf("Temperature = %.4f °C\n", temperature);
}

void printSavedSamples(const char* msg, struct tempData* s){
  for (int i = 0; i < s->numberOfSamplesRd; i++){
    printf("%s Samples : %d \t RAW = %ld \t %ldPRIu64 ns\n",
	   msg,
	   s->numberOfSamplesRd,
	   s->rawTempRd[i],
	   s->timeStamp[i]);
  }
}
/* sleep(5); */
  
/* for (int i = 0; i < fpdStruct.numberOfSamplesRd; i++){ */
/*   printf("FPD Samples : %d \t RAW = %ld" PRIu64 " ns\n", fpdStructData[i].numberOfSamplesRd, fpdStructData[i].rawTempRd, fpdStructData[i].timeStamp); */
/* } */
/* sleep(5); */



int wrStructToFile(const char* tempUnit , const char* wrFileName, struct tempData* s ){
  FILE *fp = fopen(wrFileName, "w");   // "w" = write (overwrites file)
  double measuredTemp = 0.0;
  long   rawTemp = 0;

  if (fp == NULL) {
    perror("File open failed\n");
    printf("Error occured while opening : %s \n", wrFileName);
    return 1;
  }

  // for(int i =0; i<= s.numberOfSamplesRd; i++){
  for(int i =0; i< s->numberOfSamplesRd; i++){

    if (strcmp(tempUnit, "k") == 0) {
      measuredTemp = (s->scale * (s->rawTempRd[i] + s-> offset) / 1000.0) + 273.15;
      fprintf(fp, " %.4f K \t %" PRIu64  " ns \n", measuredTemp, s->timeStamp[i]);
    }
    else if (strcmp(tempUnit, "c") == 0) {
      measuredTemp = (s->scale * ( s->rawTempRd[i] +  s->offset) / 1000.0);
      fprintf(fp, " %.4f C \t %" PRIu64 " ns \n", measuredTemp, s->timeStamp[i]);
    }

  }
  fclose(fp);
  return 0;
}

// compiler wants static
static inline uint64_t now_in_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  // convert to sec + ns, force compiler to treat as unsigned long long, 64 bit
  return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

bool checkCedrProcess(){
  FILE *cmd = popen("pgrep cedr", "r"); // read process as file
  if (!cmd) {perror("popen, cedr process not found"); return false;}
  else return true;
  // don't need pin point the PID
  /*
    int pid;
    //while (fscanf(cmd, "%d", &pid) == 1) {// cedr process is cound
    // printf("Found PID = %d\n", pid);
    if (fscanf(cmd, "%d", &pid) == 1) { return true; } // cedr process is cound
    else return false;
    }
  */
}

int main(int argc, char *argv[]){
  //double lpdTemperature = lpdScale * (lpdRaw + lpdOffset) / 1000.0+273.15;
 
  ece506TempRdState  rdTempState     = INIT; // Beginning  of synchronous state
  tempMeasuremntType measurementType = IDLE_TEMP_MEASUREMNT; // set defualt to idle cpu temp measuremnt

  int loopCntr = 0;

  const char* tempConversionType     = argv[1];
  const char* lpdTempFileWrBaseName  = "_lpd_measured_temp.txt";
  const char* fpdTempFileWrBaseName  = "_fpd_measured_temp.txt";
  const char* experimentType         = argv[2];

  size_t lpdFileStrLen    = strlen(lpdTempFileWrBaseName);
  size_t fpdFileStrLen    = strlen(fpdTempFileWrBaseName);
  size_t experimentStrLen = strlen(experimentType);

  const char* lpdNewFileWrName = malloc (lpdFileStrLen + experimentStrLen + 1);
  const char* fpdNewFileWrName = malloc (fpdFileStrLen + experimentStrLen + 1);

  // make new name for wr file
  strcpy(lpdNewFileWrName, experimentType);
  strcat(lpdNewFileWrName, lpdTempFileWrBaseName);

  // make new name for wr file
  strcpy(fpdNewFileWrName, experimentType);
  strcat(fpdNewFileWrName, fpdTempFileWrBaseName);

  // if can't malloc, return failure
  if (!lpdNewFileWrName && !fpdNewFileWrName){printf("malloc() failed\n"); return 1;}
  else{
    if (DEBUG_PRINT) {printf("LPD values will be saved in : %s \n", lpdNewFileWrName); printf("FPD values will be saved in : %s \n", fpdNewFileWrName);}
    while(loopCntr <= MAX_SAMPLES){
      switch (rdTempState){

      case (INIT) :
	rdConsts(&lpdStructData, &fpdStructData);
	if (DEBUG_PRINT) {printf("At INIT State, Offest and Scalor is collected\n");}
        if ((strcmp(argv[3], "idle") == 0)){measurementType = IDLE_TEMP_MEASUREMNT;} else{measurementType = CEDR_RUN_TIME_MEASUREMENT;}	
	rdTempState = MEASURE_LPD_TEMP;
	break;

      case (MEASURE_LPD_TEMP) :
	lpdStructData.rawTempRd[loopCntr]  = read_long_frm_file("/sys/bus/iio/devices/iio:device0/in_temp7_raw");
	lpdStructData.numberOfSamplesRd    = loopCntr;
	lpdStructData.timeStamp[loopCntr]  = now_in_ns();
	rdTempState = MEASURE_FPD_TEMP;
	break;

      case (MEASURE_FPD_TEMP) :
	fpdStructData.rawTempRd[loopCntr]   = read_long_frm_file("/sys/bus/iio/devices/iio:device0/in_temp8_raw");
	fpdStructData.numberOfSamplesRd     = loopCntr;
	fpdStructData.timeStamp[loopCntr]   = now_in_ns();
	rdTempState = SLEEP;
	break;

      case (SLEEP):
	if (DEBUG_PRINT) {printf("Sleeping..\n");}
	usleep(SLEEP_TIME);  // sleep 1 us
	rdTempState = CHECK_CEDR_PROCESS;
	break;

      case (CHECK_CEDR_PROCESS) :
	if (measurementType == IDLE_TEMP_MEASUREMNT) {
	  if (loopCntr < MAX_SAMPLES){ // keep measuring until Bucket is full
	    rdTempState = MEASURE_LPD_TEMP;
	    loopCntr++;
	  }
	  else {rdTempState = SAVE_SAMPLES_IN_FILE;}
	  if (DEBUG_PRINT) {printf("Measuring IDLE Board Temp \t samples collected : %d \n", loopCntr);}
	} 
	else { // cedr runtime measurement
	  if (checkCedrProcess()){ // cedr process is active
	    rdTempState = MEASURE_LPD_TEMP;
	    loopCntr++;
	    if (DEBUG_PRINT) {printf("CEDR process active\n");};
	  }
	  else{rdTempState = SAVE_SAMPLES_IN_FILE; if (DEBUG_PRINT) {printf("CEDR process inactive\n");}}
	}
	break;

      case RD_SAVED_SAMPLES:
	printSavedSamples("LPD Domain", &lpdStructData);
	printSavedSamples("FPD Domain", &fpdStructData);
	rdTempState = SAVE_SAMPLES_IN_FILE;
	break;

      case  SAVE_SAMPLES_IN_FILE:
	if (DEBUG_PRINT) {printf("converting into %s", tempConversionType);}
	wrStructToFile(tempConversionType, lpdNewFileWrName, &lpdStructData);
	wrStructToFile(tempConversionType, fpdNewFileWrName, &fpdStructData);
	return 0;
      }
    }
  }
  return 0;
}
