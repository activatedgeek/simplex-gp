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

// #define DEBUG

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
float_type variance(at::Tensor coeffs){
    int k = coeffs.size(0);
    auto coeffs_iter = coeffs.accessor<float_type,1>();
    float_type mom0 = 0;
    float_type mom1 = 0.;
    float_type mom2 = 0.;
    for (int i=0; i<k; ++i){
        float_type c = coeffs_iter[i];
        mom0 += c;
        mom1 += i*c;
        mom2 += i*i*c;
    }
    float_type mean = mom1/mom0;
    float_type var = mom2/mom0-mean*mean;
    //printf("%.6f",var);
    return var;
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
    static at::Tensor filter(at::Tensor src, at::Tensor ref, at::Tensor coeffs){
        int n = src.size(0);
        int srcChannels = src.size(1);
        assert(n == ref.size(0));
        int refChannels = ref.size(1);
        //int k = coeffs.size(0);
        

        // Splat into the lattice
        #ifdef DEBUG
        auto start_ts = Clock::now();
        #endif

        PermutohedralLattice lattice(refChannels, srcChannels, n, coeffs);

        float_type* arr_ref = new float_type[n * refChannels];
        float_type* arr_src = new float_type[n * srcChannels];
        // float_type* coeffs = new float_type[nk];
        auto ref_iter = ref.accessor<float_type, 2>();
        auto src_iter = src.accessor<float_type, 2>();
        //auto coeffs_iter = coeffs.accessor<float_type,1>();

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
        // for (int64_t i = 0; i < k; ++i){
        //         coeffs_src[i] = coeffs_iter[i];
        // }
        for (int i = 0; i < n; ++i){
            lattice.splat(arr_ref + i * refChannels, arr_src + i * srcChannels);
        }
        
        #ifdef DEBUG
        auto elapsed_ts = NANO_CAST(Clock::now() - start_ts).count();
        std::cout << "Hash table size: " << lattice.hashTable.size() << std::endl;
        std::cout << "Splat: " << elapsed_ts << " ns" << std::endl;
        #endif

        // Blur
        #ifdef DEBUG
        start_ts = Clock::now();
        #endif

        lattice.blur(coeffs);
        
        #ifdef DEBUG
        elapsed_ts = NANO_CAST(Clock::now() - start_ts).count();
        std::cout << "Blur: " << elapsed_ts << " ns" << std::endl;
        #endif

        // Slice from the lattice
        #ifdef DEBUG
        start_ts = Clock::now();
        #endif

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

        #ifdef DEBUG
        elapsed_ts = NANO_CAST(Clock::now() - start_ts).count();
        std::cout << "Slice: " << elapsed_ts << " ns" << std::endl;
        #endif

        at::Tensor output = torch::from_blob(outArray, {n, srcChannels}, arr_deleter).to(AT_FLOAT_TYPE);
        return output;
    }
    /* Constructor
            *         d_ : dimensionality of key vectors
            *        vd_ : dimensionality of value vectors
            * nData_ : number of points in the input
            */
    PermutohedralLattice(int d_, int vd_, int nData_, at::Tensor coeffs) : d(d_), vd(vd_), nData(nData_), hashTable(d_, vd_)
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
            float_type sigma_blur = variance(coeffs);//binomial_variance(order);
            scaleFactor[i] *= (d + 1) * sqrtf(sigma_blur+1.0f/6.0f);//sqrtf(2.0 / 3);
        }

    }

    /* Performs splatting with given position and value vectors */
    void splat(float_type *position, float_type *value) {

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
    void blur(at::Tensor coeffs){
        // Prepare arrays
        short *neighbor = new short[d + 1];
        float_type *newValue = new float_type[vd * hashTable.size()];
        float_type *oldValue = hashTable.getValues();
        float_type *hashTableBase = oldValue;
        memset(newValue, 0.0f, sizeof(float_type) * vd * hashTable.size());

        float_type *zero = new float_type[vd];
        for (int k = 0; k < vd; k++)
            zero[k] = 0;
        auto coeffs_accessor = coeffs.accessor<float_type,1>();
        // For each of d+1 axes,
        for (int j = 0; j <= d; j++){
            //printf(" %d", j);
            //fflush(stdout);

            // For each vertex in the lattice,
            for (int i = 0; i < hashTable.size(); i++){// blur point i in dimension j
                                                                                                     
                short *key = hashTable.getKeys() + i * (d); // keys to current vertex

                float_type *newVal = newValue + i * vd;
                for (int k = 0; k < vd; k++) newVal[k] = 0;
                int k = coeffs.size(0);
                int order = k/2;
                for (int nid=-order; nid<=order; ++nid){

                    for (int k = 0; k < d; k++) neighbor[k] = key[k] - nid;
                    neighbor[j] = key[j] + nid*d;

                    float_type* val = hashTable.lookup(neighbor,false);
                    val = val?val-hashTableBase+oldValue:zero;
                    float_type c = coeffs_accessor[nid+order];
                    // printf("%.3f\n",c);
                    for (int k = 0; k < vd; k++) newVal[k] += c*val[k];
                }
            }

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
