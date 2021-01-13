#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>
#include <THC/THCAtomics.cuh>

using at::Tensor;
typedef std::chrono::high_resolution_clock Clock;

// #define DEBUG

#define NANO_CAST(d) std::chrono::duration_cast<std::chrono::nanoseconds>(d)
#define BLOCK_SIZE 1024
#define PTAccessor(T,dim) at::PackedTensorAccessor32<T,dim,at::RestrictPtrTraits>
#define Ten2PTAccessor(T,x,dim) x.packed_accessor32<T,dim,at::RestrictPtrTraits>()
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
  scalar_t* bufferValues;

  /**
   * Each point has at most (pd + 1) neighbors.
   * Each entry then maps to the lattice point.
   **/
  int* entry2nid;
  uint8_t* uqentry;
public:
  size_t N, pd, vd, capacity;
  #ifdef DEBUG
  int64_t* M; // for actual size after splat.
  #endif

  HashTableGPU(size_t pd_, size_t vd_, size_t N_): 
    pd(pd_), vd(vd_), N(N_) {
    capacity = N * (pd + 1);

    gpuErrchk(cudaMallocManaged(&keys, capacity * pd * sizeof(int16_t)));

    gpuErrchk(cudaMallocManaged(&values, capacity * vd * sizeof(scalar_t)));
    gpuErrchk(cudaMallocManaged(&bufferValues, capacity * vd * sizeof(scalar_t)));
    for (size_t i = 0; i < capacity * vd; ++i) {
      values[i] = static_cast<scalar_t>(0.0);
      bufferValues[i] = static_cast<scalar_t>(0.0);
    }

    gpuErrchk(cudaMallocManaged(&entry2nid, capacity * sizeof(int)));
    gpuErrchk(cudaMallocManaged(&uqentry, capacity * sizeof(uint8_t)));
    for (size_t i = 0; i < capacity; ++i) {
      entry2nid[i] = static_cast<int>(-1);
      uqentry[i] = static_cast<uint8_t>(0);
    }

    #ifdef DEBUG
    gpuErrchk(cudaMallocManaged(&M, sizeof(int64_t)));
    *M = static_cast<int64_t>(0);
    #endif
  }

  /**
   * NOTE: Copy constructor in kernel arguments triggers the destructor.
   * Instead, manually free this in the PermutohedralLatticeGPU class.
   **/
  void free() {
    gpuErrchk(cudaFree(keys));
    gpuErrchk(cudaFree(values));
    gpuErrchk(cudaFree(bufferValues));
    gpuErrchk(cudaFree(entry2nid));
    gpuErrchk(cudaFree(uqentry));
  }

  __device__ __forceinline__ int* getEntries() { return entry2nid; }

  __device__ __forceinline__ uint8_t* getUqEntries() { return uqentry; }

  __device__ __forceinline__ int16_t* getKey(const size_t h) {
    if (entry2nid[h] == -1) {
      return nullptr;
    }
    return &keys[entry2nid[h] * pd];
  }

  __device__ __forceinline__ scalar_t* getValue(const size_t h) {
    if (entry2nid[h] == -1) {
      return nullptr;
    }
    return &values[entry2nid[h] * vd];
  }

  __device__ __forceinline__ scalar_t* getBufferValue(const size_t h) {
    if (entry2nid[h] == -1) {
      return nullptr;
    }
    return &bufferValues[entry2nid[h] * vd];
  }

  void swapBuffer() {
    scalar_t* tmp = values;
    values = bufferValues;
    bufferValues = tmp;
    for (size_t i = 0; i < capacity * vd; ++i) {
      bufferValues[i] = static_cast<scalar_t>(0.0);
    }
  }

  __device__ __forceinline__ size_t modhash(const int16_t* key) {
    size_t k = 0;
    for (uint16_t i = 0; i < pd; ++i) {
      k += static_cast<size_t>(key[i]);
      k *= static_cast<size_t>(2531011);
    }
    return k % capacity;
  }

  __device__ size_t insert(const int16_t* key, const int nid) {
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

  __device__ int get(const int16_t* key) {
    size_t h = modhash(key);
    bool loop = false;

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
        return h;
      }

      ++h;
      if (h == capacity) {
        h = 0;
        // Linear probe finished.
        if (loop) {
          return -1;
        }
        loop = true;
      }
    }
  }
};

template <typename scalar_t>
__global__ void splat_kernel(
    const PTAccessor(scalar_t,2) ref,
    PTAccessor(scalar_t,2) matE,
    PTAccessor(int16_t,2) matY,
    PTAccessor(int16_t,2) matR,
    PTAccessor(scalar_t,2) matB,
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
  auto pos = ref[n];
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

    replay[nid].entry = table.insert(key, nid);
    replay[nid].weight = bary[r];
  }
}

template <typename scalar_t>
__global__ void process_hashtable_kernel(
  HashTableGPU<scalar_t> table) {
  /**
   * NOTE: Hash table may have duplicate entries because
   * linear probing is not atomic. Assign every entry to
   * the first key match to correct this.
   **/  
  const size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= table.N) {
    return;
  }
  const size_t pd = table.pd;
  // const size_t r = blockIdx.y;
  int* entries = table.getEntries();
  uint8_t* uqentries = table.getUqEntries();
  
  for (size_t r = 0; r <= pd; ++r) {
    const size_t nid = n * (pd + 1) + r;

    if (entries[nid] >= 0) {
      auto h = table.get(table.getKey(nid));

      // Every element not re-assigned is unique.
      if (entries[nid] == entries[h]) {
        uqentries[nid] = static_cast<uint8_t>(1);
        #ifdef DEBUG
        gpuAtomicAdd(table.M, static_cast<int64_t>(1));
        #endif
      }

      entries[nid] = entries[h];
    }
  }
}

template <typename scalar_t>
__global__ void splat_value_kernel(
    const PTAccessor(scalar_t,2) src,
    HashTableGPU<scalar_t> table,
    ReplayEntry<scalar_t>* replay) {
  
  const size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= table.N) {
    return;
  }
  const size_t pd = table.pd;
  const size_t vd = src.size(1);
  // const size_t r = blockIdx.y;
  auto value = src[n];
  
  for (size_t r = 0; r <= pd; ++r) {
    const size_t nid = n * (pd + 1) + r;
    scalar_t* val = table.getValue(replay[nid].entry);
    for (size_t i = 0; i < vd; ++i) {
      gpuAtomicAdd(&val[i], replay[nid].weight * value[i]);
    }
  }
}

template <typename scalar_t>
__global__ void blur_kernel(
    HashTableGPU<scalar_t> table,
    const size_t ax,
    const PTAccessor(scalar_t,1) coeffs,
    int16_t* neighbors,
    const scalar_t* zero) {
  const size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= table.N) {
    return;
  }
  const size_t pd = table.pd;
  const size_t vd = table.vd;
  const uint8_t* uqentries = table.getUqEntries();
  int16_t* neighbor = &neighbors[n * (pd + 1)];

  for (size_t r = 0; r <= pd; ++r) {
    const size_t nid = n * (pd + 1) + r;
    if (!uqentries[nid]) {
      // Must have been processed at its lattice node.
      continue;
    }

    const int16_t* key = table.getKey(nid);
    scalar_t* bufferVal = table.getBufferValue(nid);
    const int16_t order = static_cast<int16_t>(coeffs.size(0) / 2);

    for (int16_t o = -order; o <= order; ++o) {
      for (size_t p = 0; p < pd; ++p) {
        neighbor[p] = key[p] - o;
      }
      neighbor[ax] = key[ax] + o * pd;

      int h = table.get(neighbor);
      const scalar_t* val = h >= 0 ? table.getValue(h) : zero;
      for (size_t v = 0; v < vd; ++v) {
        bufferVal[v] += coeffs[o + order] * val[v];
      }
    }
  }
}

template <typename scalar_t>
__global__ void slice_kernel(
    PTAccessor(scalar_t,2) res,
    HashTableGPU<scalar_t> table,
    ReplayEntry<scalar_t>* replay) {
  const size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= res.size(0)) {
    return;
  }

  const size_t pd = table.pd;
  const size_t vd = res.size(1);
  auto out = res[n];
  /** NOTE: magic? scaling constant. **/
  const scalar_t scale = 1.0 + pow(2.0, - (scalar_t) pd);

  for (size_t r = 0; r <= pd; ++r) {
    size_t nid = n * (pd + 1) + r;
    scalar_t* val = table.getValue(replay[nid].entry);
    for (size_t j = 0; j < vd; ++j) {
      out[j] += (replay[nid].weight * val[j]) / scale;
    }
  }
}

template <typename scalar_t>
class PermutohedralLatticeGPU {
private:
  uint16_t pd, vd;
  size_t N, order;
  scalar_t* scaleFactor;
  int16_t* canonical;
  HashTableGPU<scalar_t> hashTable;
  ReplayEntry<scalar_t>* replay;
public:
  PermutohedralLatticeGPU(uint16_t pd_, uint16_t vd_, size_t N_, const scalar_t filter_var): 
    pd(pd_), vd(vd_), N(N_), hashTable(HashTableGPU<scalar_t>(pd_, vd_, N_)) {
    
    scalar_t invStdDev = (pd + 1) * sqrt(filter_var +  static_cast<scalar_t>(1.0 / 6.0));

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
    Tensor _matE = torch::zeros(TenSize2D(N, pd + 1), TenOptType(ref.dtype(),ref.device()));
    Tensor _matY = torch::zeros(TenSize2D(N, pd + 1), TenOptType(torch::kI16,ref.device()));
    Tensor _matR = torch::zeros(TenSize2D(N, pd + 1), TenOptType(torch::kI16,ref.device()));
    Tensor _matB = torch::zeros(TenSize2D(N, pd + 2), TenOptType(ref.dtype(),ref.device()));
    int16_t* _matK = nullptr;
    gpuErrchk(cudaMallocManaged(&_matK, N * pd * sizeof(int16_t)));

    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    splat_kernel<scalar_t><<<blocks, threads>>>(
      Ten2PTAccessor(scalar_t,ref,2),
      Ten2PTAccessor(scalar_t,_matE,2), Ten2PTAccessor(int16_t,_matY,2),
      Ten2PTAccessor(int16_t,_matR,2), Ten2PTAccessor(scalar_t,_matB,2),
      _matK,
      scaleFactor, canonical,
      hashTable, replay);
    gpuErrchk(cudaPeekAtLastError());

    // const dim3 proc_blocks((N + threads.x - 1) / threads.x, pd + 1);

    process_hashtable_kernel<scalar_t><<<blocks,threads>>>(hashTable);
    gpuErrchk(cudaPeekAtLastError());

    splat_value_kernel<scalar_t><<<blocks,threads>>>(
      Ten2PTAccessor(scalar_t,src,2),
      hashTable, replay);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaFree(_matK));
  }

  void blur(const Tensor coeffs) {
    int16_t* _matNeK;
    gpuErrchk(cudaMallocManaged(&_matNeK, N * (pd + 1) * sizeof(int16_t)));

    scalar_t* zero;
    gpuErrchk(cudaMallocManaged(&zero, vd * sizeof(scalar_t)));
    for (size_t d = 0; d < vd; ++d) {
      zero[d] = static_cast<scalar_t>(0.0);
    }

    std::vector<size_t> dims;
    for (size_t d = 0; d <= pd; ++d) {
      dims.push_back(d);
    }
    /** TODO: **/
    // std::random_shuffle(dims.begin(), dims.end());

    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    for (size_t ax: dims) {
      blur_kernel<scalar_t><<<blocks, threads>>>(
        hashTable, ax, Ten2PTAccessor(scalar_t,coeffs,1),
        _matNeK, zero);
      gpuErrchk(cudaPeekAtLastError());

      gpuErrchk(cudaDeviceSynchronize());
      hashTable.swapBuffer();
    }

    gpuErrchk(cudaFree(zero));
    gpuErrchk(cudaFree(_matNeK));
  }

  Tensor slice(Tensor src, Tensor ref) {
    Tensor res = torch::zeros(TenSize2D(N, vd), TenOptType(src.dtype(),src.device()));

    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    slice_kernel<scalar_t><<<blocks, threads>>>(
      Ten2PTAccessor(scalar_t,res,2),
      hashTable, replay);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaDeviceSynchronize());
    return res;
  }

  Tensor filter(Tensor src, Tensor ref, const Tensor coeffs) {
    #ifdef DEBUG
    auto start_ts = Clock::now();
    #endif

    splat(src, ref);

    #ifdef DEBUG
    auto elapsed_ts = NANO_CAST(Clock::now() - start_ts).count();
    std::cout << "Hash table size: " << *hashTable.M << std::endl;
    std::cout << "Splat: " << elapsed_ts << " ns" << std::endl;
    #endif

    c10::cuda::CUDACachingAllocator::emptyCache();

    #ifdef DEBUG
    start_ts = Clock::now();
    #endif

    blur(coeffs);

    #ifdef DEBUG
    elapsed_ts = NANO_CAST(Clock::now() - start_ts).count();
    std::cout << "Blur: " << elapsed_ts << " ns" << std::endl;
    #endif

    #ifdef DEBUG
    start_ts = Clock::now();
    #endif

    auto res = slice(src, ref);

    #ifdef DEBUG
    elapsed_ts = NANO_CAST(Clock::now() - start_ts).count();
    std::cout << "Slice: " << elapsed_ts << " ns" << std::endl;
    #endif

    return res;
  }
};

template <typename scalar_t>
scalar_t variance(Tensor coeffs){
  auto coeffs_iter = coeffs.accessor<scalar_t,1>();

  scalar_t mom0 = static_cast<scalar_t>(0.0);
  scalar_t mom1 = static_cast<scalar_t>(0.0);
  scalar_t mom2 = static_cast<scalar_t>(0.0);

  int16_t k = static_cast<int16_t>(coeffs.size(0));
  for (int16_t i = 0; i < k; ++i) {
    scalar_t c = coeffs_iter[i];
    mom0 += c;
    mom1 += i * c;
    mom2 += i * i * c;
  }
  scalar_t mean = mom1 / mom0;
  scalar_t var = mom2 / mom0 - mean * mean;
  return var;
}

Tensor permutohedral_cuda_filter(Tensor src, Tensor ref, const Tensor coeffs) {
  Tensor out;

  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "permutohedral_lattice", ([&]{
    const Tensor cpu_coeffs = coeffs.to(torch::kCPU, false);
    const scalar_t filter_var = variance<scalar_t>(cpu_coeffs);
    PermutohedralLatticeGPU<scalar_t> lattice(ref.size(-1), src.size(-1),
                                              src.size(0), filter_var);
    out = lattice.filter(src, ref, coeffs);
  }));

  return out;
}
