#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 512

// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
// number of real galaxies
int    NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
// number of simulated random galaxies
int    NoofSim;

unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int *d_histogram;

__host__ __device__ float calculateAngle(float asc1, float decl1, float asc2, float decl2) {
  float asc1_rad, decl1_rad, asc2_rad, decl2_rad;
  float cosine;
  float angle_rad;

  asc1_rad = asc1 / 60 * M_PI / 180;
  asc2_rad = asc2 / 60 * M_PI / 180;
  decl1_rad = decl1 / 60 * M_PI / 180;
  decl2_rad = decl2 / 60 * M_PI / 180;

  cosine = sinf(decl1_rad)*sinf(decl2_rad) + cosf(decl1_rad)*cosf(decl2_rad)*cosf(asc1_rad-asc2_rad);
  if (cosine > 1.0) {
    cosine = 1.0;
  } else if (cosine < -1.0) {
    cosine = -1.0;
  }

  angle_rad = acosf(cosine);
  return angle_rad * 180 / M_PI; // angle in degrees
}

__global__ void CalculateHistogram(float* ra_1, float* decl_1, float* ra_2, float* decl_2, unsigned int* histogram, int N) {
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  int i = threadId / N;
  int j = threadId % N;
  if (i >= N) return;

  float angle = calculateAngle(ra_1[i], decl_1[i], ra_2[j], decl_2[j]);
  if (i == 0 && j == 114) {
    printf("%d,%d: (%f, %f) (%f, %f) angle %f\n", i, j, ra_1[i], decl_1[i], ra_2[j], decl_2[j], angle);
  }
  int angleIndex = (int)(angle / 0.25);
  atomicAdd(&histogram[angleIndex], 1);
}

int main(int argc, char *argv[])
{
  //int    i;
  int    noofblocks;
  int    readdata(char *argv1, char *argv2);
  int    getDevice(int deviceno);
  long int histogramDRsum, histogramDDsum, histogramRRsum;
  double w;
  double start, end, kerneltime;
  struct timeval _ttime;
  struct timezone _tzone;
  /* cudaError_t myError; */
  void CalculateHistogramCPU();
  FILE *outfil;

  if ( argc != 4 ) {printf("Usage: a.out real_data random_data output_data\n");return(-1);}

  /* if ( getDevice(0) != 0 ) return(-1); */

  if ( readdata(argv[1], argv[2]) != 0 ) return(-1);

  kerneltime = 0.0;
  gettimeofday(&_ttime, &_tzone);
  start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

  histogramDD = (unsigned int *)calloc(totaldegrees*binsperdegree, sizeof(unsigned int));
  histogramRR = (unsigned int *)calloc(totaldegrees*binsperdegree, sizeof(unsigned int));
  histogramDR = (unsigned int *)calloc(totaldegrees*binsperdegree, sizeof(unsigned int));

  // allocate memory on the GPU
  float *ra_1, *decl_1, *ra_2, *decl_2;
  // We're assuming NoofReal == NoofSim
  size_t inputSize = NoofReal*sizeof(float);
  size_t histogramSize = totaldegrees*binsperdegree*sizeof(unsigned int);

  cudaMalloc(&ra_1, inputSize);
  cudaMalloc(&decl_1, inputSize);
  cudaMalloc(&ra_2, inputSize);
  cudaMalloc(&decl_2, inputSize);
  cudaMalloc(&d_histogram, histogramSize);

  // copy data to the GPU
  cudaMemcpy(ra_1, ra_real, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(decl_1, decl_real, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(ra_2, ra_real, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(decl_2, decl_real, inputSize, cudaMemcpyHostToDevice);
  cudaMemset(d_histogram, 0, histogramSize);

  // run the kernels on the GPU
  noofblocks = (NoofReal*NoofReal + threadsperblock - 1) / threadsperblock;
  CalculateHistogram<<<noofblocks, threadsperblock>>>(ra_1, decl_1, ra_2, decl_2, d_histogram, NoofReal);

  // copy the results back to the CPU
  cudaMemcpy(histogramDD, d_histogram, histogramSize, cudaMemcpyDeviceToHost);
  // rinse and repeat for RR and DR
  cudaMemcpy(ra_1, ra_sim, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(decl_1, decl_sim, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(ra_2, ra_sim, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(decl_2, decl_sim, inputSize, cudaMemcpyHostToDevice);
  cudaMemset(d_histogram, 0, histogramSize);
  CalculateHistogram<<<noofblocks, threadsperblock>>>(ra_1, decl_1, ra_2, decl_2, d_histogram, NoofReal);
  cudaMemcpy(histogramRR, d_histogram, histogramSize, cudaMemcpyDeviceToHost);

  cudaMemcpy(ra_1, ra_real, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(decl_1, decl_real, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(ra_2, ra_sim, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(decl_2, decl_sim, inputSize, cudaMemcpyHostToDevice);
  cudaMemset(d_histogram, 0, histogramSize);
  CalculateHistogram<<<noofblocks, threadsperblock>>>(ra_1, decl_1, ra_2, decl_2, d_histogram, NoofReal);
  cudaMemcpy(histogramDR, d_histogram, histogramSize, cudaMemcpyDeviceToHost);

  cudaFree(ra_1);
  cudaFree(decl_1);
  cudaFree(ra_2);
  cudaFree(decl_2);
  cudaFree(d_histogram);

  /* CalculateHistogramCPU(); */

  gettimeofday(&_ttime, &_tzone);
  end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
  kerneltime += end-start;

  printf("time: %f s\n", kerneltime);

  // calculate omega values on the CPU
  histogramDDsum = 0;
  histogramRRsum = 0;
  histogramDRsum = 0;
  outfil = fopen(argv[3], "w");
  fprintf(outfil, "bin start\tomega\t\thist_DD\thist_DR\thist_RR\n");
  for (int n = 0; n < totaldegrees*binsperdegree; n++) {
    if (histogramRR[n] == 0) break;
    w = int(histogramDD[n] - 2*histogramDR[n] + histogramRR[n]) / float(histogramRR[n]);
    histogramDDsum += histogramDD[n];
    histogramDRsum += histogramDR[n];
    histogramRRsum += histogramRR[n];
    fprintf(outfil, "%f\t%f\t%d\t%d\t%d\n", n*0.25, w, histogramDD[n], histogramDR[n], histogramRR[n]);
  }
  printf("count sum:\n");
  printf("histogram DD: %ld, histogram DR: %ld, histogram RR: %ld\n", histogramDDsum, histogramDRsum, histogramRRsum);
  fclose(outfil);
  free(histogramDD);
  free(histogramRR);
  free(histogramDR);

  return(0);
}

int readdata(char *argv1, char *argv2)
{
  int i, linecount;
  char inbuf[180];
  double ra, dec;
  FILE *infil;

  printf("   Assuming input data is given in arc minutes!\n");
  // spherical coordinates phi and theta:
  // phi   = ra/60.0 * dpi/180.0;
  // theta = (90.0-dec/60.0)*dpi/180.0;

  /* dpi = acos(-1.0); */
  infil = fopen(argv1,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv1);return(-1);}

  // read the number of galaxies in the input file
  int announcednumber;
  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv1);return(-1);}
  linecount =0;
  while ( fgets(inbuf,180,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv1, linecount);
  else 
    {
      printf("   %s does not contain %d galaxies but %d\n",argv1, announcednumber,linecount);
      return(-1);
    }

  NoofReal = linecount;
  ra_real   = (float *)calloc(NoofReal,sizeof(float));
  decl_real = (float *)calloc(NoofReal,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i = 0;
  while ( fgets(inbuf,80,infil) != NULL )
    {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
        {
          printf("   Cannot read line %d in %s\n",i+1,argv1);
          fclose(infil);
          return(-1);
        }
      ra_real[i]   = (float)ra;
      decl_real[i] = (float)dec;
      ++i;
    }

  fclose(infil);

  if ( i != NoofReal ) 
    {
      printf("   Cannot read %s correctly\n",argv1);
      return(-1);
    }

  infil = fopen(argv2,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv2);return(-1);}

  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv2);return(-1);}
  linecount =0;
  while ( fgets(inbuf,80,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv2, linecount);
  else
    {
      printf("   %s does not contain %d galaxies but %d\n",argv2, announcednumber,linecount);
      return(-1);
    }

  NoofSim = linecount;
  ra_sim   = (float *)calloc(NoofSim,sizeof(float));
  decl_sim = (float *)calloc(NoofSim,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i =0;
  while ( fgets(inbuf,80,infil) != NULL )
    {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
        {
          printf("   Cannot read line %d in %s\n",i+1,argv2);
          fclose(infil);
          return(-1);
        }
      ra_sim[i]   = (float)ra;
      decl_sim[i] = (float)dec;
      ++i;
    }

  fclose(infil);

  if ( i != NoofSim ) 
    {
      printf("   Cannot read %s correctly\n",argv2);
      return(-1);
    }

  return(0);
}

int getDevice(int deviceNo)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("      Device %s                  device %d\n", deviceProp.name,device);
    printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
    printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
    printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
    printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
    printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
    printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
    printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
    printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
    printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
    printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
    printf("         maxGridSize                   =   %d x %d x %d\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("         concurrentKernels             =   ");
    if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
    printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
    if(deviceProp.deviceOverlap == 1)
      printf("            Concurrently copy memory/execute kernel\n");
  }

  cudaSetDevice(deviceNo);
  cudaGetDevice(&device);
  if ( device != 0 ) printf("   Unable to set device 0, using %d instead",device);
  else printf("   Using CUDA device %d\n\n", device);

  return(0);
}

void CalculateHistogramCPU() {
  float angle;
  int angleIndex;
  for (int i = 0; i < 10000; i++) {
    for (int j = i; j < 10000; j++) {
      angle = calculateAngle(ra_real[i], decl_real[i], ra_real[j], decl_real[j]);
      angleIndex = (int)(angle / 0.25);
      histogramDD[angleIndex] += 1;
    }
  }

  for (int i = 0; i < 10000; i++) {
    for (int j = i; j < 10000; j++) {
      angle = calculateAngle(ra_sim[i], decl_sim[i], ra_sim[j], decl_sim[j]);
      angleIndex = (int)(angle / 0.25);
      histogramRR[angleIndex] += 1;
    }
  }

  for (int i = 0; i < 10000; i++) {
    for (int j = i; j < 10000; j++) {
      angle = calculateAngle(ra_real[i], decl_real[i], ra_sim[j], decl_sim[j]);
      angleIndex = (int)(angle / 0.25);
      histogramDR[angleIndex] += 1;
    }
  }
}
