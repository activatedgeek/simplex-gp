#ifndef PERMUTOHEDRAL_LATTICE_H
#define PERMUTOHEDRAL_LATTICE_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <torch/torch.h>

using namespace std;

typedef float float_type;
typedef std::chrono::high_resolution_clock Clock;

#define AT_FLOAT_TYPE torch::kFloat32
#define NANO_CAST(d) std::chrono::duration_cast<std::chrono::nanoseconds>(d)

/***************************************************************/
/* Hash table implementation for permutohedral lattice
 * 
 * The lattice points are stored sparsely using a hash table.
 * The key for each point is its spatial location in the (d+1)-
 * dimensional space.
 */
/***************************************************************/
class HashTablePermutohedral{
public:
    /* Constructor
     *    kd_: the dimensionality of the position vectors on the hyperplane.
     *    vd_: the dimensionality of the value vectors
     */
    HashTablePermutohedral(int kd_, int vd_) : kd(kd_), vd(vd_){
        capacity = 1 << 15;
        filled = 0;
        entries = new Entry[capacity];
        keys = new short[kd * capacity / 2];
        values = new float_type[vd * capacity / 2];
        memset(values, 0, sizeof(float_type) * vd * capacity / 2);
    }

    // Returns the number of vectors stored.
    int size() { return filled; }

    // Returns a pointer to the keys array.
    short *getKeys() { return keys; }

    // Returns a pointer to the values array.
    float_type *getValues() { return values; }

    /* Returns the index into the hash table for a given key.
     *         key: a pointer to the position vector.
     *             h: hash of the position vector.
     *    create: a flag specifying whether an entry should be created,
     *                    should an entry with the given key not found.
     */
    int lookupOffset(short *key, size_t h, bool create = true){

        // Double hash table size if necessary
        if (filled >= (capacity / 2) - 1){
            grow();
        }

        // Find the entry with the given key
        while (1) {
            Entry e = entries[h];
            // check if the cell is empty
            if (e.keyIdx == -1){
                if (!create)
                    return -1; // Return not found.
                // need to create an entry. Store the given key.
                for (int i = 0; i < kd; i++)
                    keys[filled * kd + i] = key[i];
                e.keyIdx = filled * kd;
                e.valueIdx = filled * vd;
                entries[h] = e;
                filled++;
                return e.valueIdx;
            }

            // check if the cell has a matching key
            bool match = true;
            for (int i = 0; i < kd && match; i++)
                match = keys[e.keyIdx + i] == key[i];
            if (match)
                return e.valueIdx;

            // increment the bucket with wraparound
            h++;
            if (h == capacity)
                h = 0;
        }
    }
    ~HashTablePermutohedral(){
        delete[] entries;
        delete[] keys;
        delete[] values;
    }
    /* Looks up the value vector associated with a given key vector.
     *                k : pointer to the key vector to be looked up.
     *     create : true if a non-existing key should be created.
     */
    float_type *lookup(short *k, bool create = true){
        size_t h = hash(k) % capacity;
        int offset = lookupOffset(k, h, create);
        if (offset < 0)
            return NULL;
        else
            return values + offset;
    };

    /* Hash function used in this implementation. A simple base conversion. */
    size_t hash(const short *key){
        size_t k = 0;
        for (int i = 0; i < kd; i++){
            k += key[i];
            k *= 2531011;
        }
        return k;
    }

private:
    /* Grows the size of the hash table */
    void grow()
    {
        //printf("Resizing hash table\n");

        size_t oldCapacity = capacity;
        capacity *= 2;

        // Migrate the value vectors.
        float_type *newValues = new float_type[vd * capacity / 2];
        memset(newValues, 0, sizeof(float_type) * vd * capacity / 2);
        memcpy(newValues, values, sizeof(float_type) * vd * filled);
        delete[] values;
        values = newValues;

        // Migrate the key vectors.
        short *newKeys = new short[kd * capacity / 2];
        memcpy(newKeys, keys, sizeof(short) * kd * filled);
        delete[] keys;
        keys = newKeys;

        Entry *newEntries = new Entry[capacity];

        // Migrate the table of indices.
        for (size_t i = 0; i < oldCapacity; i++){
            if (entries[i].keyIdx == -1)
                continue;
            size_t h = hash(keys + entries[i].keyIdx) % capacity;
            while (newEntries[h].keyIdx != -1){
                h++;
                if (h == capacity)
                    h = 0;
            }
            newEntries[h] = entries[i];
        }
        delete[] entries;
        entries = newEntries;
    }

    // Private struct for the hash table entries.
    struct Entry{
        Entry() : keyIdx(-1), valueIdx(-1) {}
        int keyIdx;
        int valueIdx;
    };

    short *keys;
    float_type *values;
    Entry *entries;
    size_t capacity, filled;
    int kd, vd;
};

void arr_deleter(void *obj){

    if (obj != NULL){
        delete [] obj;
        //printf("Array object deleted :)");
    }
}


int binomial_coefficients_table[6][6] = {
    {1,0,0,0,0,0},
    {2,1,0,0,0,0},
    {6,4,1,0,0,0},
    {20,15,6,1,0,0},
    {70,56,28,8,1,0},
    {252,210,120,45,10,1},
};
float_type binomial_coefficients(int order, int k){
    assert(order<6);
    float_type normalization = float_type(binomial_coefficients_table[order][0]);//(2.0**(2*order));
    return float_type(binomial_coefficients_table[order][k>0?k:-k])/normalization;
}
float_type binomial_variance(int order){
    assert(order<6);
    return float_type(order) / 2.0f;
}

// float_type binomial_coefficients_table[4][6] = {
//     {1.0, 1.5565E-02, 5.8693E-08, 5.3618E-17, 1.1867E-29, 0.0000E+00},
//     {1.0000E+00, 9.6180E-02, 8.5571E-05, 7.0427E-10, 5.3618E-17, 3.7762E-26},
//     {1.0000e+00, 3.5321e-01, 1.5565e-02, 8.5571e-05, 5.8693e-08, 5.0224e-12},
//     {1.0000, 0.7709, 0.3532, 0.0962, 0.0156, 0.0015},
// };

// float_type binomial_coefficients(int order,int k){
//     assert (order<4);
//     float_type normalization = float_type(binomial_coefficients_table[order][0]);//(2.0**(2*order));
//     return float_type(binomial_coefficients_table[order][k>0?k:-k])/normalization;
// }
// float_type binomial_variance(int order){
//     assert (order<4);
//     return pow(float_type(order)/2.0f,2);
// }



/***************************************************************/
/* The algorithm class that performs the filter
 * 
 * PermutohedralLattice::filter(...) does all the work.
 *
 */
/***************************************************************/
class PermutohedralLattice{
public:
    /* Filters given image against a reference image.
     *     src : image to be bilateral-filtered.
     *                expected to have shape n x c
     *                where n is the number of pixels, c channels
     *    ref : reference image whose edges are to be respected.
     *                expected to have shape n x k
     *                where k is the number of features used for nlms
     *     out : output of filtering src by ref
     *                expected to have shape n x c
     */
    static at::Tensor filter(at::Tensor src, at::Tensor ref, int order=1){
        int n = src.size(0);
        int srcChannels = src.size(1);
        assert(n == ref.size(0));
        int refChannels = ref.size(1);

        PermutohedralLattice lattice(refChannels, srcChannels, n, order);

        // Splat into the lattice
        auto start_ts = Clock::now();

        float_type *arr_ref = new float_type[n * refChannels];
        float_type *arr_src = new float_type[n * srcChannels];
        auto ref_iter = ref.accessor<float_type, 2>();
        auto src_iter = src.accessor<float_type, 2>();

        for (int64_t i = 0; i < n; ++i){
            for (int64_t c = 0; c < refChannels; ++c){
                arr_ref[i * refChannels + c] = ref_iter[i][c];
            }
        }
        for (int64_t i = 0; i < n; ++i){
            for (int64_t c = 0; c < srcChannels; ++c){
                arr_src[i * srcChannels + c] = src_iter[i][c];
            }
        }

        for (int i = 0; i < n; ++i){
            lattice.splat(arr_ref + i * refChannels, arr_src + i * srcChannels);
        }

        lattice.all_splat_ts = NANO_CAST(Clock::now() - start_ts).count();

        std::cout << "Hash table filled " << lattice.hashTable.size() << std::endl;

        // // Old code
        // float_type *col = new float_type[im.channels+1];
        // col[im.channels] = 1; // homogeneous coordinate

        // float_type *imPtr = im(0, 0, 0);
        // float_type *refPtr = ref(0, 0, 0);
        // for (int t = 0; t < im.frames; t++) {
        //     for (int y = 0; y < im.height; y++) {
        //         for (int x = 0; x < im.width; x++) {
        //             for (int c = 0; c < im.channels; c++) {
        //                 col[c] = *imPtr++;
        //             }
        //             lattice.splat(refPtr, col);
        //             refPtr += ref.channels;
        //         }
        //     }
        // }

        // Blur
        start_ts = Clock::now();

        lattice.blur(order);
        
        lattice.blur_ts = NANO_CAST(Clock::now() - start_ts).count();

        // Slice from the lattice
        start_ts = Clock::now();

        lattice.beginSlice();
        float_type *outArray = new float_type[n * srcChannels];
        for (int i = 0; i < n * srcChannels; ++i){
            outArray[i] = 0;
        }
        for (int i = 0; i < n; ++i){
            float_type *col = outArray + i * srcChannels;
            lattice.slice(col);
        }
        delete[] arr_ref;
        delete[] arr_src;

        lattice.slice_ts = NANO_CAST(Clock::now() - start_ts).count();

        at::Tensor output = torch::from_blob(outArray, {n, srcChannels}, arr_deleter).to(AT_FLOAT_TYPE);
        at::TensorAccessor<float_type, 2> fa = output.accessor<float_type, 2>();
        //printf("%.6f",fa[0][0]);
        // at::Tensor output = at::empty({n,srcChannels});
        // auto out_iter = output.accessor<float_type,2>();
        // for (int64_t i=0; i<n; ++i){
        //     for (int64_t c=0; c<srcChannels;++c){
        //         out_iter[i][c] = outArray[i+c*srcChannels];
        //     }
        // }
        //delete [] outArray;
        // delete[] outArray;
        // Need to figure out how to fill out with outArray

        // lattice.beginSlice();
        // float_type *outPtr = out(0, 0, 0);
        // for (int t = 0; t < im.frames; t++) {
        //     for (int y = 0; y < im.height; y++) {
        //         for (int x = 0; x < im.width; x++) {
        //             lattice.slice(col);
        //             float_type scale = 1.0f/col[im.channels];
        //             for (int c = 0; c < im.channels; c++) {
        //                 *outPtr++ = col[c]*scale;
        //             }
        //         }
        //     }
        // }

        size_t total = lattice.all_splat_ts + lattice.blur_ts + lattice.slice_ts;

        std::cout << "Init: " << lattice.init_ts << " ns\n";
        std::cout << "All Splat: " << lattice.all_splat_ts << " ns, " << (float_type(lattice.all_splat_ts) / total) << "\n";
        std::cout << "Avg Hashmap/Splat Ratio: " << lattice.avg_ratio_ts << "\n";
        std::cout << "Blur: " << lattice.blur_ts << " ns, " << (float_type(lattice.blur_ts) / total) << "\n";
        std::cout << "Slice: " << lattice.slice_ts << " ns, " << (float_type(lattice.slice_ts) / total) << "\n";

        return output;
    }
    /* Constructor
            *         d_ : dimensionality of key vectors
            *        vd_ : dimensionality of value vectors
            * nData_ : number of points in the input
            */
    PermutohedralLattice(int d_, int vd_, int nData_, int order) : d(d_), vd(vd_), nData(nData_), hashTable(d_, vd_)
    {

        // Allocate storage for various arrays
        elevated = new float_type[d + 1];
        scaleFactor = new float_type[d];

        greedy = new short[d + 1];
        rank = new char[d + 1];
        barycentric = new float_type[d + 2];
        replay = new ReplayEntry[nData * (d + 1)];
        nReplay = 0;
        canonical = new short[(d + 1) * (d + 1)];
        key = new short[d + 1];

        auto start_ts = Clock::now();
        
        // compute the coordinates of the canonical simplex, in which
        // the difference between a contained point and the zero
        // remainder vertex is always in ascending order. (See pg.4 of paper.)
        for (int i = 0; i <= d; i++){
            for (int j = 0; j <= d - i; j++)
                canonical[i * (d + 1) + j] = i;
            for (int j = d - i + 1; j <= d; j++)
                canonical[i * (d + 1) + j] = i - (d + 1);
        }

        // Compute parts of the rotation matrix E. (See pg.4-5 of paper.)
        for (int i = 0; i < d; i++){
            // the diagonal entries for normalization
            scaleFactor[i] = 1.0f / (sqrtf((float_type)(i + 1) * (i + 2)));

            /* We presume that the user would like to do a Gaussian blur of standard deviation
                * 1 in each dimension (or a total variance of d, summed over dimensions.)
                * Because the total variance of the blur performed by this algorithm is not d,
                * we must scale the space to offset this.
                *
                * The total variance of the algorithm is (See pg.6 and 10 of paper):
                *    [variance of splatting] + [variance of blurring] + [variance of splicing]
                *     = d(d+1)(d+1)/12 + d(d+1)(d+1)/2 + d(d+1)(d+1)/12
                *     = 2d(d+1)(d+1)/3.
                *
                * So we need to scale the space by (d+1)sqrt(2/3).
                */
            float_type sigma_blur = binomial_variance(order);
            scaleFactor[i] *= (d + 1) * sqrtf(sigma_blur+1.0f/6.0f);//sqrtf(2.0 / 3);
        }

        init_ts = NANO_CAST(Clock::now() - start_ts).count();
    }

    /* Performs splatting with given position and value vectors */
    void splat(float_type *position, float_type *value) {
        auto start_ts = Clock::now();

        // first rotate position into the (d+1)-dimensional hyperplane
        elevated[d] = -d * position[d - 1] * scaleFactor[d - 1];
        for (int i = d - 1; i > 0; i--)
            elevated[i] = (elevated[i + 1] - i * position[i - 1] * scaleFactor[i - 1] +
                                         (i + 2) * position[i] * scaleFactor[i]);
        elevated[0] = elevated[1] + 2 * position[0] * scaleFactor[0];

        // prepare to find the closest lattice points
        float_type scale = 1.0f / (d + 1);
        char *myrank = rank;
        short *mygreedy = greedy;

        // greedily search for the closest zero-colored lattice point
        int sum = 0;
        for (int i = 0; i <= d; i++){
            float_type v = elevated[i] * scale;
            float_type up = ceilf(v) * (d + 1);
            float_type down = floorf(v) * (d + 1);

            if (up - elevated[i] < elevated[i] - down)
                mygreedy[i] = (short)up;
            else
                mygreedy[i] = (short)down;

            sum += mygreedy[i];
        }
        sum *= scale; //Modification here /= d+1;

        // rank differential to find the permutation between this simplex and the canonical one.
        // (See pg. 3-4 in paper.)
        memset(myrank, 0, sizeof(char) * (d + 1));
        for (int i = 0; i < d; i++)
            for (int j = i + 1; j <= d; j++)
                if (elevated[i] - mygreedy[i] < elevated[j] - mygreedy[j])
                    myrank[i]++;
                else
                    myrank[j]++;

        if (sum > 0){
            // sum too large - the point is off the hyperplane.
            // need to bring down the ones with the smallest differential
            for (int i = 0; i <= d; i++){
                if (myrank[i] >= d + 1 - sum){
                    mygreedy[i] -= d + 1;
                    myrank[i] += sum - (d + 1);
                }
                else
                    myrank[i] += sum;
            }
        }
        else if (sum < 0) {
            // sum too small - the point is off the hyperplane
            // need to bring up the ones with largest differential
            for (int i = 0; i <= d; i++){
                if (myrank[i] < -sum){
                    mygreedy[i] += d + 1;
                    myrank[i] += (d + 1) + sum;
                }else
                    myrank[i] += sum;
            }
        }

        // Compute barycentric coordinates (See pg.10 of paper.)
        memset(barycentric, 0, sizeof(float_type) * (d + 2));
        for (int i = 0; i <= d; i++){
            barycentric[d - myrank[i]] += (elevated[i] - mygreedy[i]) * scale;
            barycentric[d + 1 - myrank[i]] -= (elevated[i] - mygreedy[i]) * scale;
        }
        barycentric[0] += 1.0f + barycentric[d + 1];

        auto start_hash_ts = Clock::now();

        // Splat the value into each vertex of the simplex, with barycentric weights.
        for (int remainder = 0; remainder <= d; remainder++){
            // Compute the location of the lattice point explicitly (all but the last coordinate - it's redundant because they sum to zero)
            for (int i = 0; i < d; i++)
                key[i] = mygreedy[i] + canonical[remainder * (d + 1) + myrank[i]];

            // Retrieve pointer to the value at this vertex.
            float_type *val = hashTable.lookup(key, true);

            // Accumulate values with barycentric weight.
            //auto value_a = value.accessor<float_type,1>();
            for (int i = 0; i < vd; i++)
                val[i] += (barycentric[remainder] * value[i]); // This should be an averaging operation rather than sum perhaps?

            // Record this interaction to use later when slicing
            replay[nReplay].offset = val - hashTable.getValues(); // pointer arithmetic
            replay[nReplay].weight = barycentric[remainder];
            nReplay++;
        }

        auto end_ts = Clock::now();

        auto splat_ts = NANO_CAST(end_ts - start_ts);
        auto hash_ts = NANO_CAST(end_ts - start_hash_ts);

        float_type ratio_ts = float_type(hash_ts.count()) / splat_ts.count();
        avg_ratio_ts += float_type(ratio_ts - avg_ratio_ts) / nReplay;
    }

    // Prepare for slicing
    void beginSlice(){
        nReplay = 0;
    }

    /* Performs slicing out of position vectors. Note that the barycentric weights and the simplex
     * containing each position vector were calculated and stored in the splatting step.
     * We may reuse this to accelerate the algorithm. (See pg. 6 in paper.)
     */
    void slice(float_type *col){
        float_type *base = hashTable.getValues();
        for (int j = 0; j < vd; j++)
            col[j] = 0; // Zero the output channels for the current pixel (col)

        for (int i = 0; i <= d; i++){// Loop over the input channels (d+1 simplex neighbors)
                                                                                
            ReplayEntry r = replay[nReplay++]; // get the pointer offset and weight for this pixel
            for (int j = 0; j < vd; j++){  // Loop over the output channels
                                                                                                                                       
                col[j] += r.weight * base[r.offset + j] / (1 + powf(2, -d)); //; // add to channel j (of the pixel associated with col)
            }  // magic scaling constant from krahenbuhls implementation?                                                                                                                            
        }
    }

    /* Performs a Gaussian blur along each projected axis in the hyperplane. */
    void blur(int order){
        // Prepare arrays
        short *neighbor = new short[d + 1];
        short *neighbor1 = new short[d + 1];
        short *neighbor2 = new short[d + 1];
        float_type *newValue = new float_type[vd * hashTable.size()];
        float_type *oldValue = hashTable.getValues();
        float_type *hashTableBase = oldValue;
        memset(newValue, 0.0f, sizeof(float_type) * vd * hashTable.size());

        float_type *zero = new float_type[vd];
        for (int k = 0; k < vd; k++)
            zero[k] = 0;

        // For each of d+1 axes,
        for (int j = 0; j <= d; j++){
            //printf(" %d", j);
            //fflush(stdout);

            // For each vertex in the lattice,
            for (int i = 0; i < hashTable.size(); i++){// blur point i in dimension j
                                                                                                     
                short *key = hashTable.getKeys() + i * (d); // keys to current vertex

                //  // Init neighbor to (order) below value along given axis
                float_type *newVal = newValue + i * vd;
                for (int k = 0; k < vd; k++) newVal[k] = 0;

                for (int nid=-order; nid<=order; ++nid){

                    for (int k = 0; k < d; k++) neighbor[k] = key[k] - nid;
                    neighbor[j] = key[j] + nid*d;

                    float_type* val = hashTable.lookup(neighbor,false);
                    val = val?val-hashTableBase+oldValue:zero;
                    float_type c = binomial_coefficients(order,nid);
                    // printf("%.3f\n",c);
                    for (int k = 0; k < vd; k++) newVal[k] += c*val[k];
                }
                // for (int k = 0; k < d; k++){// shouldn't it be k<d+1?
                //     neighbor1[k] = key[k] + 1;
                //     neighbor2[k] = key[k] - 1;
                // }
                // neighbor1[j] = key[j] - d;
                // neighbor2[j] = key[j] + d; // keys to the neighbors along the given axis.

                // float_type *oldVal = oldValue + i * vd;
                // float_type *newVal = newValue + i * vd;

                // float_type *vm1, *vp1;

                // vm1 = hashTable.lookup(neighbor1, false); // look up first neighbor
                // if (vm1)
                //     vm1 = vm1 - hashTableBase + oldValue;
                // else
                //     vm1 = zero;

                // vp1 = hashTable.lookup(neighbor2, false); // look up second neighbor
                // if (vp1)
                //     vp1 = vp1 - hashTableBase + oldValue;
                // else
                //     vp1 = zero;

                // // Mix values of the three vertices
                // for (int k = 0; k < vd; k++)
                //     newVal[k] = (0.5f * vm1[k] + 1.0f * oldVal[k] + 0.5f * vp1[k]); // factor of two from krahenbuhl's implementation
            
            }                                                                             // because the gaussians should not be normalized

            float_type *tmp = newValue;
            newValue = oldValue;
            oldValue = tmp;
            // the freshest data is now in oldValue, and newValue is ready to be written over
        }

        // depending where we ended up, we may have to copy data
        if (oldValue != hashTableBase){
            // assert(false);
            memcpy(hashTableBase, oldValue, hashTable.size() * vd * sizeof(float_type));
            delete oldValue;
        }
        else{
            //assert(false);
            delete newValue;
        }
        //printf("\n");

        delete zero;
        delete neighbor;
        delete neighbor1;
        delete neighbor2;
    }

private:
    int d, vd, nData;
    float_type *elevated, *scaleFactor, *barycentric;
    short *canonical;
    short *key;

    // slicing is done by replaying splatting (ie storing the sparse matrix)
    struct ReplayEntry{
        int offset;
        float_type weight;
    } * replay;
    int nReplay, nReplaySub;

    // runtime bookkeeping
    float_type avg_ratio_ts = 0.0f;
    size_t init_ts;
    size_t all_splat_ts;
    size_t blur_ts;
    size_t slice_ts;

public:
    char *rank;
    short *greedy;
    HashTablePermutohedral hashTable;
    ~PermutohedralLattice(){
        delete[] replay;
        delete[] canonical;
        delete[] elevated;
        delete[] scaleFactor;
        delete[] greedy;
        delete[] rank;
        delete[] barycentric;
        delete[] key;
    };
};
#endif
