#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cuda_runtime.h>
namespace{
  constexpr int kBlocksize=256;
  constexpr int kItemsPerThread=4;

  __device__ __forceinline__ float warp_reduce_sum(float v){
    #pragma unroll
    for(int offset=16;offset>0;offset>>=1){
      v+=__shfl_down_sync(0xfffffffu,v,offset);
    }
    return v;
  }

  __device__ __forceinline__ float warp_reduce_max(float v){
    #pragma unroll
    for(int offset=16;offset>0;offset>>=1){
      v=fmaxf(v,__shfl_down_sync(0xfffffffu,v,offset));
    }
    return v;
  }

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float v){
  __shared__ float warp_sums[BLOCK_SIZE/32];
  int lane=threadIdx.x&31;
  int warp=threadIdx.x>>5;
  v=warp_reduce_sum(v);
  if(lane==0){
    warp_sums[warp]=v;
  }
  __syncthreads();
  v=(threadIdx.x<BLOCK_SIZE/32)?warp_sums[lane]:0.0f;
  if(warp==0){
    v=warp_reduce_sum(v);
  }
  return v;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_max(float v){
  __shared__ float warp_maxes[BLOCK_SIZE/32];
  int lane=threadIdx.x&31;
  int warp=threadIdx.x>>5;
  v=warp_reduce_max(v);
  if(lane==0){
    warp_maxes[warp]=v;
  }
  __syncthreads();
  v=(threadIdx.x<BLOCK_SIZE/32)?warp_maxes[lane]:-INT_MAX;
  if(warp==0){
    v=warp_reduce_max(v);
  }
  return v;
}

__global__ void max_first(const float* __restrict__ input,float* __restrict__ partial_max,int n){
  int tid=threadIdx.x;
  int base=(blockIdx.x*blockDim.x+tid)*kItemsPerThread;
  int stride=gridDim.x*blockDim.x*kItemsPerThread;
  float local_max=-INT_MAX;
  for(int i=base;i<n;i+=stride){
    if(i+3<n){
      float4 v=*reinterpret_cast<const float4*>(input+i);
      local_max=fmaxf(local_max,v.x);
      local_max=fmaxf(local_max,v.y);
      local_max=fmaxf(local_max,v.z);
      local_max=fmaxf(local_max,v.w);
    }else{
      for(int j=0;j<kItemsPerThread;j++){
        int idx=i+j;
        if(idx<n){
          local_max=fmaxf(local_max,input[idx]);
        }
      }
    }
  }

  local_max=block_reduce_max<kBlocksize>(local_max);
  if(tid==0){
    partial_max[blockIdx.x]=local_max;
  }
}

__global__ void final_max(const float* __restrict__ partial_max,float*__restrict__ max_value,int n){
  int tid=threadIdx.x;
  float local_max=-INT_MAX;
  for(int i=tid;i<n;i+=blockDim.x){
    local_max=fmaxf(local_max,partial_max[i]);
  }
  local_max=block_reduce_max<kBlocksize>(local_max);
  if(tid==0){
    *max_value=local_max;
  }

}
__global__ void  exp_sum(const float *__restrict__ input,
                               float *__restrict__ output,
                               float *__restrict__ partial_sum,
                               const float *__restrict__ max_value, 
                               int n)
{
  int tid=threadIdx.x;
  int base=(blockIdx.x*blockDim.x+tid)*kItemsPerThread;
  int stride=gridDim.x*blockDim.x*kItemsPerThread;
  float max_v=*max_value;
  float local_sum=0.0f;
  for(int i=base;i<n;i+=stride){
    if(i+3<n){
      float4 v=*reinterpret_cast<const float4*>(input+i);
      float4 e;
      e.x=__expf(v.x-max_v);
      e.y=__expf(v.y-max_v);
      e.z=__expf(v.z-max_v);
      e.w=__expf(v.w-max_v);
      *reinterpret_cast<float4*>(output+i)=e;
      local_sum=e.x+e.y+e.z+e.w;
    }else{
      #pragma unroll
      for(int j=0;j<kItemsPerThread;j++){
        int idx=i+j;
        if(idx<n){
          float e=__expf(input[idx]-max_v);
          output[idx]=e;
          local_sum+=e;
        }
      }
    }
  }

  local_sum=block_reduce_sum<kBlocksize>(local_sum);
  if(tid==0){
    partial_sum[blockIdx.x]=local_sum;
  }
}

__global__ void final_sum(const float *__restrict__ partial_sum,
                                      float *__restrict__ sum_value, int n)
{
  int tid=threadIdx.x;
  int local_sum=0.0f;
  for(int i=tid;i<n;i+=blockDim.x){
    local_sum+=partial_sum[i];
  }
  local_sum=block_reduce_sum<kBlocksize>(local_sum);
  if(tid==0){
    *sum_value=local_sum;
  }
}

__global__ void normalize(float *__restrict__ output,
                                 const float *__restrict__ sum_value, int n)
{
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  int stride=gridDim.x*blockDim.x;
  float inv_sum=1/(*sum_value);
  for(int i=tid;i<n;i+stride){
    output[i]*=inv_sum;
  }
}
}