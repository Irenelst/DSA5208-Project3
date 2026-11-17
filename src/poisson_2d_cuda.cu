#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr double PI = 3.14159265358979323846;
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){std::cerr<<"CUDA error "<<__FILE__<<":"<<__LINE__<<": "<<cudaGetErrorString(e)<<"\n"; std::exit(2);} } while(0)

inline size_t idx(int i,int j,int s){return size_t(i)*s+j;}

__global__ void jacobi_kernel(const double* old_u,double* new_u,const double* f,int N,int s,double h2){
    int j=blockIdx.x*blockDim.x+threadIdx.x+1;
    int i=blockIdx.y*blockDim.y+threadIdx.y+1;
    if(i<=N && j<=N){
        size_t k=size_t(i)*s+j;
        new_u[k]=0.25*(old_u[k-s]+old_u[k+s]+old_u[k-1]+old_u[k+1]-h2*f[k]);
    }
}

void init_source(std::vector<double>& f,int N,int s,double h){
    for(int i=1;i<=N;++i) for(int j=1;j<=N;++j)
        f[idx(i,j,s)]=-2.0*PI*PI*std::sin(PI*i*h)*std::sin(PI*j*h);
}

double relative_residual(const std::vector<double>& u,const std::vector<double>& f,int N,int s,double h){
    double r2=0,f2=0,h2=h*h;
    for(int i=1;i<=N;++i)for(int j=1;j<=N;++j){
        size_t k=idx(i,j,s);
        double Au=(u[k-s]+u[k+s]+u[k-1]+u[k+1]-4*u[k])/h2;
        double r=f[k]-Au;
        r2+=r*r;f2+=f[k]*f[k];
    }
    return std::sqrt(r2/f2);
}

double max_analytical_error(const std::vector<double>& u,int N,int s,double h){
    double emax=0;
    for(int i=1;i<=N;++i)for(int j=1;j<=N;++j)
        emax=std::max(emax,std::abs(u[idx(i,j,s)]-std::sin(PI*i*h)*std::sin(PI*j*h)));
    return emax;
}

int main(int argc,char** argv){
    if(argc<3 || argc>4){std::cerr<<"Usage: "<<argv[0]<<" validation N | benchmark N 1000\n";return 1;}
    std::string mode=argv[1];
    int N=0,K=0;
    try{N=std::stoi(argv[2]); if(mode=="benchmark" && argc==4)K=std::stoi(argv[3]);}
    catch(...){std::cerr<<"Invalid integer argument\n";return 1;}
    if(N<=0 || (mode=="benchmark" && (argc!=4 || K!=1000)) || (mode=="validation" && argc!=3) || (mode!="validation" && mode!="benchmark")){
        std::cerr<<"Use validation N or benchmark N 1000\n";return 1;
    }
    int s=N+2;
    size_t cells=size_t(s)*s,bytes=cells*sizeof(double);
    double h=1.0/(N+1);
    std::vector<double> u(cells,0.0),f(cells,0.0);
    init_source(f,N,s,h);
    int count=0;CUDA_CHECK(cudaGetDeviceCount(&count));
    if(count<1){std::cerr<<"No CUDA GPU allocated\n";return 2;}
    cudaDeviceProp prop;CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    double *d_old=nullptr,*d_new=nullptr,*d_f=nullptr;
    auto e2e_start=std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMalloc(&d_old,bytes));CUDA_CHECK(cudaMalloc(&d_new,bytes));CUDA_CHECK(cudaMalloc(&d_f,bytes));
    CUDA_CHECK(cudaMemset(d_old,0,bytes));CUDA_CHECK(cudaMemset(d_new,0,bytes));
    CUDA_CHECK(cudaMemcpy(d_f,f.data(),bytes,cudaMemcpyHostToDevice));
    dim3 block(16,16),grid((N+block.x-1)/block.x,(N+block.y-1)/block.y);
    auto step=[&](){jacobi_kernel<<<grid,block>>>(d_old,d_new,d_f,N,s,h*h);CUDA_CHECK(cudaGetLastError());std::swap(d_old,d_new);};
    std::cout<<std::setprecision(12)<<"GPU = "<<prop.name<<"\n";
    if(mode=="validation"){
        const int max_iter=100000,frequency=100;
        double residual=relative_residual(u,f,N,s,h);
        int iter=0;
        while(iter<max_iter && residual>=1e-6){
            for(int k=0;k<frequency && iter<max_iter;++k,++iter)step();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(u.data(),d_old,bytes,cudaMemcpyDeviceToHost));
            residual=relative_residual(u,f,N,s,h);
        }
        double error=max_analytical_error(u,N,s,h);
        std::cout<<"Mode = validation\nN = "<<N<<"\nIterations = "<<iter<<"\nFinal residual = "<<residual<<"\nMax analytical error = "<<error<<"\n";
        std::cout<<"CSV,validation,2D,"<<N<<","<<iter<<","<<residual<<","<<error<<"\n";
        CUDA_CHECK(cudaFree(d_old));CUDA_CHECK(cudaFree(d_new));CUDA_CHECK(cudaFree(d_f));
        return residual<1e-6?0:3;
    }
    // CUDA context and allocations are initialized before the kernel-only timer.
    cudaEvent_t start,stop;CUDA_CHECK(cudaEventCreate(&start));CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for(int k=0;k<K;++k)step();
    CUDA_CHECK(cudaEventRecord(stop));CUDA_CHECK(cudaEventSynchronize(stop));
    float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
    CUDA_CHECK(cudaMemcpy(u.data(),d_old,bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_old));CUDA_CHECK(cudaFree(d_new));CUDA_CHECK(cudaFree(d_f));
    auto e2e_end=std::chrono::steady_clock::now();
    double residual=relative_residual(u,f,N,s,h);
    double compute_s=ms/1000.0;
    double e2e_s=std::chrono::duration<double>(e2e_end-e2e_start).count();
    std::cout<<"Mode = benchmark\nN = "<<N<<"\nK = "<<K<<"\nGPU compute time (s) = "<<compute_s<<"\nEnd-to-end time (s) = "<<e2e_s<<"\nFinal residual = "<<residual<<"\n";
    std::cout<<"CSV,benchmark,2D,"<<N<<","<<K<<","<<compute_s<<","<<e2e_s<<","<<residual<<"\n";
    return 0;
}
