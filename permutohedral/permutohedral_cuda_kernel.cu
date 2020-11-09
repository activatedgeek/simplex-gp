#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <THC/THCAtomics.cuh>

using at::Tensor;
typedef std::chrono::high_resolution_clock Clock;

#define NANO_CAST(d) std::chrono::duration_cast<std::chrono::nanoseconds>(d)
#define BLOCK_SIZE 1024
#define PTAccessor2D(T) at::PackedTensorAccessor32<T,2,at::RestrictPtrTraits>
#define Accessor1Di(T) at::TensorAccessor<T,1,at::RestrictPtrTraits,int32_t>
#define Ten2PTAccessor2D(T, x) x.packed_accessor32<T,2,at::RestrictPtrTraits>()
#define TenSize2D(m,n) {static_cast<int64_t>(m), static_cast<int64_t>(n)}
#define TenOptType(T, D) torch::dtype(T).device(D.type(),D.index())

// https://stackoverflow.com/a/14038590/2425365
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename scalar_t>
struct ReplayEntry{
  size_t entry;
  scalar_t weight;
};

template <typename scalar_t>
class HashTableGPU {
private:
  int16_t* keys;
  scalar_t* values;

  /**
   * Each point has at most (pd + 1) neighbors.
   * Each entry then maps to the lattice point.
   **/
  int* entry2nid;
public:
  size_t N, pd, vd, capacity;

  HashTableGPU(size_t pd_, size_t vd_, size_t N_): 
    pd(pd_), vd(vd_), N(N_) {
    capacity = N * (pd + 1);

    gpuErrchk(cudaMallocManaged(&keys, capacity * pd * sizeof(int16_t)));

    gpuErrchk(cudaMallocManaged(&values, capacity * vd * sizeof(scalar_t)));
    for (size_t i = 0; i < capacity * vd; ++i) {
      values[i] = static_cast<scalar_t>(0.0);
    }

    gpuErrchk(cudaMallocManaged(&entry2nid, capacity * sizeof(int)));
    for (size_t i = 0; i < capacity; ++i) {
      entry2nid[i] = static_cast<int>(-1);
    }
  }

  /**
   * NOTE: Copy constructor in kernel arguments triggers the destructor.
   * Instead, manually free this in the PermutohedralLatticeGPU class.
   **/
  void free() {
    gpuErrchk(cudaFree(keys));
    gpuErrchk(cudaFree(values));
    gpuErrchk(cudaFree(entry2nid));
  }

  __host__ __device__ int* getEntries() { return entry2nid; }

  __device__ int16_t* getKeys() { return keys; }

  __device__ scalar_t* getValues() { return values; }

  __device__ int16_t* lookupKey(int nid) {
    return &keys[nid * pd];
  }

  __device__ scalar_t* lookupValue(size_t h) {
    return &values[entry2nid[h] * vd];
  }

  __device__ __forceinline__ size_t modhash(int16_t* key) {
    size_t k = 0;
    for (uint16_t i = 0; i < pd; ++i) {
      k += static_cast<size_t>(key[i]);
      k *= static_cast<size_t>(2531011);
    }
    return k % capacity;
  }

  __device__ size_t insert(int16_t* key, int nid) {
    size_t h = modhash(key);

    while (true) {
      int cas = atomicCAS(&entry2nid[h], -1, -2); // Returns the (old) value at location.

      if (cas == -2) { // Locked by another thread.
      } else if (cas == -1) { // Lock acquired.
        for (size_t i = 0; i < pd; ++i) {
          keys[nid * pd + i] = key[i];
        }

        atomicExch(&entry2nid[h], nid);

        return h;
      } else { // Otherwise check if an existing key matches.
        bool match = true;
        for (size_t i = 0; i < pd && match; ++i) {
          match = keys[cas * pd + i] == key[i];
        }
        if (match) {
          return h;
        }
      }

      // Linear probing.
      ++h;
      if (h == capacity) {
        h = 0;
      }
    }
  }

  __device__ int get(int16_t* key) {
    size_t h = modhash(key);

    while (true) {
      int nid = entry2nid[h];
      if (nid == -1) {
        return -1;
      }

      bool match = true;
      for (size_t i = 0; i < pd && match; ++i) {
        match = keys[nid * pd + i] == key[i];
      }
      if (match) {
        return nid;
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
    int16_t* matK,
    const scalar_t* scaleFactor,
    const int16_t* canonical,
    HashTableGPU<scalar_t> table,
    ReplayEntry<scalar_t>* replay) {
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
  auto key = &matK[n * pd];

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
    size_t nid = n * (pd + 1) + r;

    for (uint16_t i = 0; i < pd; ++i) {
      key[i] = y[i] + canonical[r * (pd + 1) + rank[i]];
    }

    table.insert(key, nid);

    /**
     * TODO: 
     **/
    // size_t h = table.insert(key, nid);
    // replay[nid].entry = h;
    // replay[nid].weight = bary[r];
    
    // scalar_t* val = table.lookupValue(h);
    // for (uint16_t i = 0; i < vd; ++i) {
    //   gpuAtomicAdd(&val[i], bary[r] * value[i]);
    // }
    // printf("%lld (key = [%d], hash = %lld)\n", nid, key[0], table.modhash(key));
  }
}

template <typename scalar_t>
__global__ void process_hashtable_kernel(
    HashTableGPU<scalar_t> table) {
  
  const size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= table.N) {
    return;
  }
  const size_t r = blockIdx.y;
  const size_t pd = table.pd;

  const size_t id = n * (pd + 1) + r;

  /**
   * NOTE: Hash table may have duplicate entries because
   * linear probing is not atomic. Assign every entry to
   * the first key match to correct this.
   **/
  int* entries = table.getEntries();
  if (entries[id] >= 0) {
    entries[id] = table.get(table.lookupKey(entries[id]));
  }
}

template <typename scalar_t>
__global__ void blur_kernel(
    HashTableGPU<scalar_t> table) {
  /**
   * TODO: Blur kernel.
   **/
}

template <typename scalar_t>
__global__ void slice_kernel(
    PTAccessor2D(scalar_t) res,
    HashTableGPU<scalar_t> table,
    ReplayEntry<scalar_t>* replay) {
  const size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= res.size(0)) {
    return;
  }

  const uint16_t pd = table.pd;
  const uint16_t vd = res.size(1);
  auto out = res[n];

  for (uint16_t r = 0; r <= pd; ++r) {
    size_t nid = n * (pd + 1) + r;
    scalar_t* val = table.lookupValue(replay[nid].entry);
    for (uint16_t j = 0; j < vd; ++j) {
      out[j] += replay[nid].weight * val[j] / (1 + powf(2, -pd));
    }
  }
}

template <typename scalar_t>
class PermutohedralLatticeGPU {
private:
  uint16_t pd, vd;
  size_t N;
  scalar_t* scaleFactor;
  int16_t* canonical;
  HashTableGPU<scalar_t> hashTable;
  ReplayEntry<scalar_t>* replay;
public:
  PermutohedralLatticeGPU(uint16_t pd_, uint16_t vd_, size_t N_): 
    pd(pd_), vd(vd_), N(N_), hashTable(HashTableGPU<scalar_t>(pd_, vd_, N_)) {
    
    /** TODO: Adjust this scale factor for larger kernel stencils. **/
    scalar_t invStdDev = (pd + 1) * sqrt(2.0f / 3);

    gpuErrchk(cudaMallocManaged(&scaleFactor, pd * sizeof(scalar_t)));
    for (uint16_t i = 0; i < pd; ++i) {
      scaleFactor[i] = invStdDev / static_cast<scalar_t>(sqrt((i + 1) * (i + 2)));
    }

    gpuErrchk(cudaMallocManaged(&canonical, (pd + 1) * (pd + 1) * sizeof(int16_t)));
    for (uint16_t i = 0; i <= pd; ++i) {
      for (uint16_t j = 0; j <= pd - i; ++j) {
        canonical[i * (pd + 1) + j] = i;
      }
      for (uint16_t j = pd - i + 1; j <= pd; ++j) {
        canonical[i * (pd + 1) + j] = i - (pd + 1);
      }
    }

    gpuErrchk(cudaMallocManaged(&replay, N * (pd + 1) * sizeof(ReplayEntry<scalar_t>)));
  }

  ~PermutohedralLatticeGPU() {
    hashTable.free();
    gpuErrchk(cudaFree(scaleFactor));
    gpuErrchk(cudaFree(canonical));
    gpuErrchk(cudaFree(replay));
  }

  void splat(Tensor src, Tensor ref) {
    _matE = torch::zeros(TenSize2D(N, pd + 1), TenOptType(ref.dtype(),ref.device()));
    _matY = torch::zeros(TenSize2D(N, pd + 1), TenOptType(torch::kI16,ref.device()));
    _matR = torch::zeros(TenSize2D(N, pd + 1), TenOptType(torch::kI16,ref.device()));
    _matB = torch::zeros(TenSize2D(N, pd + 2), TenOptType(ref.dtype(),ref.device()));
    gpuErrchk(cudaMallocManaged(&_matK, N * pd * sizeof(int16_t)));

    const dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + threads.x - 1) / threads.x);

    splat_kernel<scalar_t><<<blocks, threads>>>(
      Ten2PTAccessor2D(scalar_t,src),
      Ten2PTAccessor2D(scalar_t,ref),
      Ten2PTAccessor2D(scalar_t,_matE),
      Ten2PTAccessor2D(int16_t,_matY),
      Ten2PTAccessor2D(int16_t,_matR),
      Ten2PTAccessor2D(scalar_t,_matB),
      _matK,
      scaleFactor,
      canonical,
      hashTable,
      replay
    );
    gpuErrchk(cudaPeekAtLastError());

    blocks.y = pd + 1;
    process_hashtable_kernel<scalar_t><<<blocks,threads>>>(
      hashTable
    );
    gpuErrchk(cudaPeekAtLastError());
  }

  void blur(uint16_t d) {
    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    blur_kernel<scalar_t><<<blocks, threads>>>(
      hashTable
    );
    gpuErrchk(cudaPeekAtLastError());
  }

  void slice(Tensor src, Tensor ref) {
    res = torch::zeros(TenSize2D(N, vd), TenOptType(src.dtype(),src.device()));

    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    slice_kernel<scalar_t><<<blocks, threads>>>(
      Ten2PTAccessor2D(scalar_t,res),
      hashTable,
      replay
    );
    gpuErrchk(cudaPeekAtLastError());
  }

  Tensor filter(Tensor src, Tensor ref) {
    splat(src, ref);

    for (uint16_t i = 0; i <= pd; ++i) {
      blur(i);
    }

    slice(src, ref);

    gpuErrchk(cudaDeviceSynchronize());

    int* entries = hashTable.getEntries();
    std::set<int> s;
    for (uint16_t i = 0; i < (N * (pd + 1)); ++i) {
      s.insert(entries[i]);
      // std::cout << entries[i] << " ";
    }
    std::cout << "Hash table size: " << s.size() << std::endl;

    return res;
  }
private:
  int16_t* _matK;
  Tensor res;
  // Matrices for internal lattice operations.
  Tensor _matE, _matY, _matR, _matB;
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
