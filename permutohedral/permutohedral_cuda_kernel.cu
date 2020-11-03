/**
 * NOTE: This block explains the comment codes used throughout.
 *  DYNALLOC: Dynamic malloc could be potentially disastrous for speed.
 *            Better patterns, contiguous shared thread memory, or pre-allocation.
 *  MALLOC: Need error checks after every CUDA malloc call.
 *  LOCKFREEHASH: Generating the next free slot by using a naive lock-free approach
 *    may cause problems when using many threads. The solution is to do
 *    post-processing, which makes code less-readable. Keeping things simple for
 *    now.
 **/

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <THC/THCAtomics.cuh>

using at::Tensor;

#define BLOCK_SIZE 1024
#define PTAccessor2D(T) at::PackedTensorAccessor32<T,2,at::RestrictPtrTraits>
#define Accessor1Di(T) at::TensorAccessor<T,1,at::RestrictPtrTraits,int32_t>
#define Ten2PTAccessor2D(T, x) x.packed_accessor32<T,2,at::RestrictPtrTraits>()
#define TenSize2D(m,n) {static_cast<int64_t>(m), static_cast<int64_t>(n)}
#define TenOptType(T, D) torch::dtype(T).device(D.type(),D.index())

template <typename scalar_t>
class HashTableGPU {
private:
  uint16_t pd, vd;
  size_t N, capacity;

  int16_t* keys;
  scalar_t* values;

  size_t* filled;

  /**
   * NOTE: CUDA atomic operations only defined on (unsigned) int types. Other
   * types need to be implemented manually, but int works for us.
   **/
  int* entryIdx;
  int* slotLock;
public:
  HashTableGPU(uint16_t pd_, uint16_t vd_, size_t N_): 
    pd(pd_), vd(vd_), N(N_) {
    capacity = N * (pd + 1); // Loose upper bound.

    cudaMallocManaged(&keys, capacity * pd * sizeof(int16_t));
    cudaMemset(keys, static_cast<int16_t>(0), capacity * pd);

    cudaMallocManaged(&values, capacity * vd * sizeof(scalar_t));
    cudaMemset(values, static_cast<scalar_t>(0.0), capacity * vd);

    cudaMallocManaged(&filled, sizeof(size_t));
    cudaMemset(filled, static_cast<size_t>(0), 1);

    cudaMallocManaged(&entryIdx, capacity * sizeof(int));
    cudaMemset(entryIdx, static_cast<int>(-1), capacity);

    cudaMallocManaged(&slotLock, sizeof(int));
    cudaMemset(slotLock, static_cast<size_t>(-1), 1);
  }

  ~HashTableGPU() {
    cudaFree(keys);
    cudaFree(values);
    cudaFree(filled);
    cudaFree(entryIdx);
    cudaFree(slotLock);
  }

  size_t size() {
    return *filled;
  }

  __device__ __forceinline__ size_t next_slot() {
    while (true) {
      int cas = atomicCAS(slotLock, static_cast<int>(-1), static_cast<int>(-2));
      if (cas == -1) { // Lock acquired.
        size_t slot = (*filled)++;
        atomicExch(slotLock, static_cast<int>(-1)); // Release lock.
        return slot;
      }
    }
  }

  __device__ __forceinline__ size_t hash(Accessor1Di(int16_t) key) {
    size_t k = 0;
    for (uint16_t i = 0; i < pd; ++i) {
      k += static_cast<size_t>(key[i]);
      k = k * static_cast<size_t>(2531011);
    }
    return k;
  }

  __device__ scalar_t* lookup(Accessor1Di(int16_t) key, bool create = false) {
    size_t h = hash(key) % capacity;

    while (true) {
      int cas = atomicCAS(&entryIdx[h],
                          static_cast<int>(-1), static_cast<int>(-2));

      if (cas == -2) { // Locked by another thread.
      } else if (cas == -1) { // Lock acquired.
        if (!create) {
          atomicExch(&entryIdx[h], static_cast<int>(-1));
          return nullptr;
        }

        /** WARN: LOCKFREEHASH **/
        size_t slot = next_slot();
        
        for (uint16_t i = 0; i < pd; ++i) {
          keys[slot * pd + i] = key[i];
        }

        atomicExch(&entryIdx[h], static_cast<int>(slot));

        return &values[slot * vd];
      } else { // No need to lock, simply return pointer if keys match.
        int slot = entryIdx[h];

        bool match = true;
        for (uint16_t i = 0; i < pd && match; ++i) {
          match = keys[slot * pd + i] == key[i];
        }
        if (match) {
          return &values[slot * vd];
        }
      }

      ++h;
      if (h == capacity) {
        h = 0;
      }
    }
  }
};

template <typename scalar_t>
__global__ void splat_kernel(
    const PTAccessor2D(scalar_t) src,
    const PTAccessor2D(scalar_t) ref,
    PTAccessor2D(scalar_t) matE,
    PTAccessor2D(int16_t) matY,
    PTAccessor2D(int16_t) matR,
    PTAccessor2D(scalar_t) matB,
    PTAccessor2D(int16_t) matK,
    const scalar_t* scaleFactor,
    const int16_t* canonical,
    HashTableGPU<scalar_t> hashTable,
    int64_t* counter
  ) {
  const size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= ref.size(0)) {
    return;
  }

  const uint16_t pd = ref.size(1);
  const uint16_t vd = src.size(1);
  auto pos = ref[n];
  auto value = src[n];
  auto elevated = matE[n];
  auto y = matY[n];
  auto rank = matR[n];
  auto bary = matB[n];
  auto key = matK[n];

  elevated[pd] = - pd * pos[pd - 1] * scaleFactor[pd - 1];
  for (uint16_t i = pd - 1; i > 0; i--) {
    elevated[i] = elevated[i + 1] - i * pos[i - 1] * scaleFactor[i - 1] +
                  (i + 2) * pos[i] * scaleFactor[i];
  }
  elevated[0] = elevated[1] + 2.0 * pos[0] * scaleFactor[0];

  int16_t h = 0;
  for (uint16_t i = 0; i <= pd; ++i) {
    y[i] = static_cast<int16_t>(round(elevated[i] / (pd + 1))) * (pd + 1);
    h += y[i];

    rank[i] = 0;
    bary[i] = 0.0;
  }
  h /= (pd + 1);

  bary[pd + 1] = 0.0;

  for (uint16_t i = 0; i < pd; ++i) {
    for (uint16_t j = i + 1; j <= pd; ++j) {
      if (elevated[i] - y[i] < elevated[j] - y[j]) {
        rank[i]++;
      } else {
        rank[j]++;
      }
    }
  }

  if (h > 0) {
    for (uint16_t i = 0; i <= pd; ++i) {
      if (rank[i] >= pd + 1 - h) {
          y[i] -= pd + 1;
          rank[i] += h - (pd + 1);
      }
      else {
        rank[i] += h;
      }
    }
  } else if (h < 0) {
    for (uint16_t i = 0; i <= pd; ++i) {
      if (rank[i] < -h) {
        y[i] += pd + 1;
        rank[i] += h + (pd + 1);
      } else {
        rank[i] += h;
      }
    }
  }

  for (uint16_t i = 0; i <= pd; ++i) {
    scalar_t delta = static_cast<scalar_t>(elevated[i] - y[i]) / (pd + 1);
    bary[pd - rank[i]] += delta;
    bary[pd + 1 - rank[i]] -= delta;
  }
  bary[0] += 1.0 + bary[pd + 1];

  for (uint16_t r = 0; r <= pd; ++r) {
    for (uint16_t i = 0; i < pd; ++i) {
      key[i] = y[i] + canonical[r * (pd + 1) + rank[i]];
    }

    scalar_t* val = hashTable.lookup(key, true);
    // for (uint16_t i = 0; i < vd; ++i) {
    //   gpuAtomicAdd(&val[i], bary[r] * value[i]);
    // }
    /** TODO: bookkeeping hash table location to re-use later. **/
  }

  /** FIXME: Only for illustrative purposes. Remove later. **/
  gpuAtomicAdd(counter, static_cast<int64_t>(1));
}

template <typename scalar_t>
class PermutohedralLatticeGPU {
private:
  uint16_t pd, vd;
  size_t N;
  scalar_t* scaleFactor;
  int16_t* canonical;
  int64_t* counter;
  HashTableGPU<scalar_t> hashTable;
public:
  PermutohedralLatticeGPU(uint16_t pd_, uint16_t vd_, size_t N_): 
    pd(pd_), vd(vd_), N(N_), hashTable(HashTableGPU<scalar_t>(pd_, vd_, N_)) {
    
    /** TODO: Adjust this scale factor for larger kernel stencils. **/
    scalar_t invStdDev = (pd + 1) * sqrt(2.0f / 3);

    /** TODO: MALLOC **/
    cudaMallocManaged(&scaleFactor, pd * sizeof(scalar_t));
    for (uint16_t i = 0; i < pd; ++i) {
      scaleFactor[i] = invStdDev / ((scalar_t) sqrt((i + 1) * (i + 2)));
    }

    /** TODO: MALLOC **/
    cudaMallocManaged(&canonical, (pd + 1) * (pd + 1) * sizeof(int16_t));
    for (uint16_t i = 0; i <= pd; ++i) {
      for (uint16_t j = 0; j <= pd - i; ++j) {
        canonical[i * (pd + 1) + j] = i;
      }
      for (uint16_t j = pd - i + 1; j <= pd; ++j) {
        canonical[i * (pd + 1) + j] = i - (pd + 1);
      }
    }

    /** FIXME: Only for illustrative purposes. Remove later. **/
    cudaMallocManaged(&counter, sizeof(int64_t));
    *counter = static_cast<int64_t>(0);
  }

  ~PermutohedralLatticeGPU() {
    cudaFree(scaleFactor);
    cudaFree(canonical);
    cudaFree(counter);
  }

  void splat(Tensor src, Tensor ref) {
    _matE = torch::zeros(TenSize2D(N, pd + 1), TenOptType(ref.dtype(),ref.device()));
    _matY = torch::zeros(TenSize2D(N, pd + 1), TenOptType(torch::kI16,ref.device()));
    _matR = torch::zeros(TenSize2D(N, pd + 1), TenOptType(torch::kI16,ref.device()));
    _matB = torch::zeros(TenSize2D(N, pd + 2), TenOptType(ref.dtype(),ref.device()));
    _matK = torch::zeros(TenSize2D(N, pd), TenOptType(torch::kI16,ref.device()));

    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    splat_kernel<scalar_t><<<blocks, threads>>>(
      Ten2PTAccessor2D(scalar_t,src),
      Ten2PTAccessor2D(scalar_t,ref),
      Ten2PTAccessor2D(scalar_t,_matE),
      Ten2PTAccessor2D(int16_t,_matY),
      Ten2PTAccessor2D(int16_t,_matR),
      Ten2PTAccessor2D(scalar_t,_matB),
      Ten2PTAccessor2D(int16_t,_matK),
      scaleFactor,
      canonical,
      hashTable,
      counter
    );
  }

  Tensor filter(Tensor src, Tensor ref) {
    splat(src, ref);

    cudaDeviceSynchronize();

    std::cout << *counter << std::endl;
    std::cout << hashTable.size() << std::endl;

    /** TODO: fixme once computations completed **/
    return at::ones_like(src);
  }
private:
  // Matrices for internal lattice operations.
  Tensor _matE, _matY, _matR, _matB, _matK;
};

Tensor permutohedral_cuda_filter(Tensor src, Tensor ref) {
  Tensor out;

  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "permutohedral_lattice", ([&]{
    PermutohedralLatticeGPU<scalar_t> lattice(ref.size(-1), src.size(-1),
                                              src.size(0));
    out = lattice.filter(src, ref);
  }));

  return out;
}
