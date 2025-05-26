#include "exp_integral.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <sys/time.h>
#include <cmath>
#include <unistd.h>
#include <cstdio>
#include <cuda_runtime.h>

using namespace std;

bool cpu = true;
bool gpu = true;
bool verbose = false;
bool timing = false;

int maxIterations=2000000000;
unsigned int n = 10, numberOfSamples = 10;
double a = 0.0, b = 10.0;

int parseArguments(int argc, char *argv[]);
void printUsage();
double get_time_sec();
void outputResults(const std::vector<float>& floatData, const std::vector<double>& doubleData);
void compareResults(const std::vector<float>& cpuF, const std::vector<float>& gpuF,
                    const std::vector<double>& cpuD, const std::vector<double>& gpuD);


int main(int argc, char* argv[]) {
    parseArguments(argc, argv);

    if (verbose) {
        cout << "n=" << n << endl;
        cout << "numberOfSamples=" << numberOfSamples << endl;
        cout << "a=" << a << endl;
        cout << "b=" << b << endl;
        cout << "timing=" << timing << endl;
        cout << "verbose=" << verbose << endl;
        cout << "cpu=" << cpu << endl;
        cout << "gpu=" << gpu << endl;
    }

    // Sanity checks
        if (a>=b) {
                cout << "Incorrect interval ("<<a<<","<<b<<") has been stated!" << endl;
                return 0;
        }
        if (n<=0) {
                cout << "Incorrect orders ("<<n<<") have been stated!" << endl;
                return 0;
        }
        if (numberOfSamples<=0) {
                cout << "Incorrect number of samples ("<<numberOfSamples<<") have been stated!" << endl;
                return 0;
        }

    std::vector<float> cpuFloat(n * numberOfSamples), gpuFloat(n * numberOfSamples);
    std::vector<double> cpuDouble(n * numberOfSamples), gpuDouble(n * numberOfSamples);

    double timeCpuFloat = 0.0, timeCpuDouble = 0.0, timeGpuFloat = 0.0, timeGpuDouble = 0.0;

    if (cpu) {

        //CPU_Flot version (calculating + timing)
        double startF = get_time_sec();
        for (unsigned int ui = 1; ui <= n; ui++) {
            for (unsigned int uj = 1; uj <= numberOfSamples; uj++) {
                double x = a + uj * (b - a) / numberOfSamples;
                cpuFloat[(ui - 1) * numberOfSamples + (uj - 1)] = exponentialIntegralFloatCPU(ui, x);
                }
        }
        double endF = get_time_sec();
        timeCpuFloat = endF - startF;

        //Cpu_Double version (calculating + timing)
         double startD = get_time_sec();
        for (unsigned int ui = 1; ui <= n; ui++) {
            for (unsigned int uj = 1; uj <= numberOfSamples; uj++) {
                double x = a + uj * (b - a) / numberOfSamples;
                cpuDouble[(ui - 1) * numberOfSamples + (uj - 1)] = exponentialIntegralDoubleCPU(ui, x);
            }
        }
        double endD = get_time_sec();
        timeCpuDouble = endD - startD;
    }

    if (gpu) {
        cudaStream_t stream1,stream2;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        
	exponentialIntegralGPUFloat(n, numberOfSamples, a, b, gpuFloat.data(), &timeGpuFloat, stream1);
        exponentialIntegralGPUDouble(n, numberOfSamples, a, b, gpuDouble.data(), &timeGpuDouble, stream2);
        
	cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        }

     if (timing) {
        if (cpu) {
            printf("CPU float computation time:   %f seconds\n", timeCpuFloat);
            printf("CPU double computation time:  %f seconds\n", timeCpuDouble);
        }
        if (gpu) {
            printf("GPU float computation time:  %f seconds\n", timeGpuFloat);
            printf("GPU double computation time: %f seconds\n", timeGpuDouble);
        }
     }

    if (cpu && gpu) {
    if (timeGpuFloat > 0) printf("Speedup (float):  %.2fx\n", timeCpuFloat / timeGpuFloat);        
    if (timeGpuDouble > 0) printf("Speedup (double): %.2fx\n", timeCpuDouble / timeGpuDouble);
    }

    if (verbose && cpu) outputResults(cpuFloat, cpuDouble);

    if (cpu && gpu) compareResults(cpuFloat, gpuFloat, cpuDouble, gpuDouble);
    return 0;
}


double get_time_sec() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void printUsage () {
        printf("exponentialIntegral program\n");
        printf("by: Jose Mauricio Refojo <refojoj@tcd.ie>\n");
        printf("This program will calculate a number of exponential integrals\n");
        printf("usage:\n");
        printf("exponentialIntegral.out [options]\n");
        printf("      -a   value   : will set the a value of the (a,b) interval in which the samples are taken to value (default: 0.0)\n");
        printf("      -b   value   : will set the b value of the (a,b) interval in which the samples are taken to value (default: 10.0)\n");
        printf("      -c           : will skip the CPU test\n");
        printf("      -g           : will skip the GPU test\n");
        printf("      -h           : will show this usage\n");
        printf("      -i   size    : will set the number of iterations to size (default: 2000000000)\n");
        printf("      -n   size    : will set the n (the order up to which we are calculating the exponential integrals) to size (default: 10)\n");
        printf("      -m   size    : will set the number of samples taken in the (a,b) interval to size (default: 10)\n");
        printf("      -t           : will output the amount of time that it took to generate each norm (default: no)\n");
        printf("      -v           : will activate the verbose mode  (default: no)\n");
        printf("     \n");
}

int parseArguments (int argc, char *argv[]) {
        int c;

        while ((c = getopt (argc, argv, "cghn:m:a:b:tv")) != -1) {
                switch(c) {
                        case 'c':
                                cpu=false; break;        //Skip the CPU test
                        case 'h':
                                printUsage(); exit(0); break;
                        case 'i':
                                maxIterations = atoi(optarg); break;
                        case 'n':
                                n = atoi(optarg); break;
                        case 'm':
                                numberOfSamples = atoi(optarg); break;
                        case 'a':
                                a = atof(optarg); break;
                        case 'b':
                                b = atof(optarg); break;
                        case 't':
                                timing = true; break;
                        case 'v':
                                verbose = true; break;
                        default:
                                fprintf(stderr, "Invalid option given\n");
                                printUsage();
                                return -1;
                }
        }
        return 0;
}

void outputResults(const std::vector<float>& floatData, const std::vector<double>& doubleData) {
    for (unsigned int i = 1; i <= n; i++) {
        for (unsigned int j = 1; j <= numberOfSamples; j++) {
            double x = a + j * (b - a) / numberOfSamples;
            cout << "CPU==> exponentialIntegralDouble(" << i << "," << x << ")="
                 << doubleData[(i - 1) * numberOfSamples + (j - 1)] << " , ";
            cout << "exponentialIntegralFloat(" << i << "," << x << ")="
                 << floatData[(i - 1) * numberOfSamples + (j - 1)] << endl;
        }
    }
}

void compareResults(const std::vector<float>& cpuF, const std::vector<float>& gpuF,
                    const std::vector<double>& cpuD, const std::vector<double>& gpuD) {
    float maxF = 0.0f; int idxF = -1;
    double maxD = 0.0; int idxD = -1;
    for (unsigned int i = 0; i < n * numberOfSamples; ++i) {
        float diffF = fabs(cpuF[i] - gpuF[i]);
        double diffD = fabs(cpuD[i] - gpuD[i]);
        if (diffF > maxF) { maxF = diffF; idxF = i; }
        if (diffD > maxD) { maxD = diffD; idxD = i; }
    }
    if (idxF >= 0) {
        printf("Max float diff = %e at (n=%d, x=%.6f)\n", maxF, idxF / numberOfSamples + 1,
               a + ((idxF % numberOfSamples) + 1) * (b - a) / numberOfSamples);
    }
    if (idxD >= 0) {
        printf("Max double diff = %e at (n=%d, x=%.6f)\n", maxD, idxD / numberOfSamples + 1,
               a + ((idxD % numberOfSamples) + 1) * (b - a) / numberOfSamples);
    }
    if (maxF > 1e-5f || maxD > 1e-5)
        printf("[WARNING] Numerical difference exceeds threshold!\n");
}
