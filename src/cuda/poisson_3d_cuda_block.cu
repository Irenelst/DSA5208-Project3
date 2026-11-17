#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

constexpr double PI = 3.14159265358979323846;
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){std::cerr<<"CUDA error "<<__FILE__<<":"<<__LINE__<<": "<<cudaGetErrorString(e)<<"\n";std::exit(2);} } while(0)

inline size_t idx3d(int i,int j,int k,int s) {
    return (static_cast<size_t>(i)*s+j)*s+k;
}

// One thread updates one interior unknown; z is the contiguous memory direction.
__global__ void jacobi_3d_kernel(const double* old_u, double* new_u,
                                 const double* f, int N, int s, double h2) {
    const int k = blockIdx.x*blockDim.x+threadIdx.x+1;
    const int j = blockIdx.y*blockDim.y+threadIdx.y+1;
    const int i = blockIdx.z*blockDim.z+threadIdx.z+1;
    if(i<=N && j<=N && k<=N) {
        const size_t pos=(static_cast<size_t>(i)*s+j)*s+k;
        const size_t plane=static_cast<size_t>(s)*s;
        new_u[pos]=(old_u[pos-plane]+old_u[pos+plane]
                   +old_u[pos-s]+old_u[pos+s]
                   +old_u[pos-1]+old_u[pos+1]
                   -h2*f[pos])/6.0;
    }
}

void init_source(std::vector<double>& f,int N,int s,double h) {
    for(int i=1;i<=N;++i) for(int j=1;j<=N;++j) for(int k=1;k<=N;++k)
        f[idx3d(i,j,k,s)]=-3.0*PI*PI*std::sin(PI*i*h)*std::sin(PI*j*h)*std::sin(PI*k*h);
}

double relative_residual(const std::vector<double>& u,const std::vector<double>& f,
                         int N,int s,double h) {
    double r2=0.0,f2=0.0;
    const double h2=h*h;
    const size_t plane=static_cast<size_t>(s)*s;
    for(int i=1;i<=N;++i) for(int j=1;j<=N;++j) for(int k=1;k<=N;++k) {
        const size_t p=idx3d(i,j,k,s);
        const double Au=(u[p-plane]+u[p+plane]+u[p-s]+u[p+s]+u[p-1]+u[p+1]-6.0*u[p])/h2;
        const double r=f[p]-Au;
        r2+=r*r;
        f2+=f[p]*f[p];
    }
    return std::sqrt(r2)/std::sqrt(f2);
}

double max_analytical_error(const std::vector<double>& u,int N,int s,double h) {
    double emax=0.0;
    for(int i=1;i<=N;++i) for(int j=1;j<=N;++j) for(int k=1;k<=N;++k) {
        const double exact=std::sin(PI*i*h)*std::sin(PI*j*h)*std::sin(PI*k*h);
        emax=std::max(emax,std::abs(u[idx3d(i,j,k,s)]-exact));
    }
    return emax;
}

int main(int argc,char** argv) {
    if(argc<3 || argc>7) {
        std::cerr<<"Usage: "<<argv[0]<<" validation N | benchmark N 1000\n";
        return 1;
    }
    std::string mode=argv[1];
    int N=0,K=0;
    try {
        size_t used=0;
        N=std::stoi(argv[2],&used);
        if(used!=std::string(argv[2]).size()) throw std::invalid_argument("N");
        if(mode=="benchmark" && (argc==4 || argc==7)) {
            K=std::stoi(argv[3],&used);
            if(used!=std::string(argv[3]).size()) throw std::invalid_argument("K");
        }
    } catch(...) {std::cerr<<"Invalid integer argument\n";return 1;}
    if(N<=0 || (mode=="benchmark" && ((argc!=4 && argc!=7) || K!=1000)) ||
       (mode=="validation" && argc!=3 && argc!=6) || (mode!="validation" && mode!="benchmark")) {
        std::cerr<<"Use validation N or benchmark N 1000\n";return 1;
    }
    int bx=8,by=8,bz=4;
    if(mode=="benchmark" && argc==7) {
        try { bx=std::stoi(argv[4]);by=std::stoi(argv[5]);bz=std::stoi(argv[6]); }
        catch(...) {std::cerr<<"Invalid block dimensions\n";return 1;}
    } else if(mode=="validation" && argc==6) {
        try {bx=std::stoi(argv[3]);by=std::stoi(argv[4]);bz=std::stoi(argv[5]);}
        catch(...) {std::cerr<<"Invalid block dimensions\n";return 1;}
    } else if(!((mode=="benchmark" && argc==4)||(mode=="validation" && argc==3))) {
        std::cerr<<"Usage: validation N [bx by bz] | benchmark N 1000 [bx by bz]\n";return 1;
    }
    if(bx<=0 || by<=0 || bz<=0 || 1LL*bx*by*bz>1024) {
        std::cerr<<"Invalid CUDA block size\n";return 1;
    }
    const size_t s=static_cast<size_t>(N)+2;
    if(s>std::numeric_limits<size_t>::max()/s || s*s>std::numeric_limits<size_t>::max()/s ||
       s*s*s>std::numeric_limits<size_t>::max()/sizeof(double)) {
        std::cerr<<"Grid size overflow\n";return 1;
    }
    const size_t cells=s*s*s,bytes=cells*sizeof(double);
    const double h=1.0/(N+1);
    std::vector<double> u(cells,0.0),f(cells,0.0);
    init_source(f,N,static_cast<int>(s),h);

    int device_count=0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if(device_count<1) {std::cerr<<"No CUDA GPU allocated\n";return 2;}
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::cout<<std::setprecision(12)<<"GPU = "<<prop.name<<"\n";
    std::cout<<"Dimension = 3D\n";

    const auto e2e_start=std::chrono::steady_clock::now();
    double *d_old=nullptr,*d_new=nullptr,*d_f=nullptr;
    CUDA_CHECK(cudaMalloc(&d_old,bytes));
    CUDA_CHECK(cudaMalloc(&d_new,bytes));
    CUDA_CHECK(cudaMalloc(&d_f,bytes));
    CUDA_CHECK(cudaMemset(d_old,0,bytes));
    CUDA_CHECK(cudaMemset(d_new,0,bytes));
    CUDA_CHECK(cudaMemcpy(d_f,f.data(),bytes,cudaMemcpyHostToDevice));

    if(bx>prop.maxThreadsDim[0] || by>prop.maxThreadsDim[1] || bz>prop.maxThreadsDim[2] ||
       1LL*bx*by*bz>prop.maxThreadsPerBlock) {
        std::cerr<<"Block size exceeds GPU limits\n";return 1;
    }
    std::cout<<"Block = "<<bx<<"x"<<by<<"x"<<bz<<"\n";
    const dim3 block(bx,by,bz);
    const dim3 grid((N+block.x-1)/block.x,(N+block.y-1)/block.y,(N+block.z-1)/block.z);
    auto step=[&]() {
        jacobi_3d_kernel<<<grid,block>>>(d_old,d_new,d_f,N,static_cast<int>(s),h*h);
        CUDA_CHECK(cudaGetLastError());
        std::swap(d_old,d_new);
    };

    if(mode=="validation") {
        const int max_iter=100000,check_frequency=100;
        int iter=0;
        double residual=relative_residual(u,f,N,static_cast<int>(s),h);
        while(iter<max_iter && residual>=1e-6) {
            for(int t=0;t<check_frequency && iter<max_iter;++t,++iter) step();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(u.data(),d_old,bytes,cudaMemcpyDeviceToHost));
            residual=relative_residual(u,f,N,static_cast<int>(s),h);
        }
        const double error=max_analytical_error(u,N,static_cast<int>(s),h);
        std::cout<<"Mode = validation\nN = "<<N<<"\nIterations = "<<iter
                 <<"\nFinal residual = "<<residual<<"\nMax analytical error = "<<error
                 <<"\nStatus = "<<(residual<1e-6?"CONVERGED":"NOT CONVERGED")<<"\n";
        std::cout<<"CSV,validation,3D,"<<N<<","<<iter<<","<<residual<<","<<error<<"\n";
        CUDA_CHECK(cudaFree(d_old)); CUDA_CHECK(cudaFree(d_new)); CUDA_CHECK(cudaFree(d_f));
        return residual<1e-6?0:3;
    }

    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for(int t=0;t<K;++t) step();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms=0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms,start,stop));
    CUDA_CHECK(cudaMemcpy(u.data(),d_old,bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_old)); CUDA_CHECK(cudaFree(d_new)); CUDA_CHECK(cudaFree(d_f));
    const auto e2e_end=std::chrono::steady_clock::now();
    const double compute_s=elapsed_ms/1000.0;
    const double e2e_s=std::chrono::duration<double>(e2e_end-e2e_start).count();
    const double residual=relative_residual(u,f,N,static_cast<int>(s),h);
    std::cout<<"Mode = benchmark\nN = "<<N<<"\nK = "<<K
             <<"\nGPU compute time (s) = "<<compute_s
             <<"\nEnd-to-end time (s) = "<<e2e_s
             <<"\nFinal residual = "<<residual<<"\n";
    std::cout<<"CSV,benchmark,3D,"<<N<<","<<K<<","<<compute_s<<","<<e2e_s<<","<<residual<<"\n";
    return 0;
}
