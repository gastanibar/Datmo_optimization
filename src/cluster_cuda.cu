#include "cluster_cuda.hpp"

const int d = 25;
const int x_row = 70;
const float step = (3.14 / 2) / d;

void memAllocate() {
    cudaMalloc((void**)&d_th, sizeof(double) * d);
    cudaMalloc((void**)&d_idmaxth, sizeof(int));
    cudaMalloc((void**)&d_x, sizeof(double) * x_row * 2); //sovradimensiono X di poco
    cudaMalloc((void**)&d_c1, sizeof(double) * d * x_row);
    cudaMalloc((void**)&d_c2, sizeof(double) * d * x_row);
    cudaMalloc((void**)&d_row, sizeof(int));
    cudaMalloc((void**)&d_max, sizeof(double) * 2 * d);
    cudaMalloc((void**)&d_min, sizeof(double) * 2 * d);
    cudaMalloc((void**)&d_c1max, sizeof(double) * d * x_row);
    cudaMalloc((void**)&d_c1min, sizeof(double) * d * x_row);
    cudaMalloc((void**)&d_c2max, sizeof(double) * d * x_row);
    cudaMalloc((void**)&d_c2min, sizeof(double) * d * x_row);
    cudaMalloc((void**)&d_b, sizeof(double) * d);
    cudaMalloc((void**)&d_coef, sizeof(double) * 12);
    //memoria pinnata
    cudaMallocHost(&h_coef, sizeof(double) * 12);
    cudaMallocHost(&h_x, sizeof(double) * x_row * 2);
    cudaStreamCreate(&s1); 
    cublasCreate(&handle);

    cublasStatus_t stat;
    stat = cublasSetStream(handle, s1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "errore nell'associazione tra handle e s1\n" << stat << std::endl;;
    }

    //indico che il puntatore deve essere passato per riferimento al device
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
}

void memFree() {
    cudaFree(d_th);
    cudaFree(d_x);
    cudaFree(d_c1);
    cudaFree(d_c2);
    cudaFree(d_row);
    cudaFree(d_max);
    cudaFree(d_min);
    cudaFree(d_c1max);
    cudaFree(d_c1min);
    cudaFree(d_c2max);
    cudaFree(d_c2min);
    cudaFree(d_b);
    cudaFree(d_coef);
    cudaFree(&d_idmaxth);
    cudaStreamDestroy(s1);
    cublasDestroy(handle);
}

/*
    operazioni atomiche per double, di per se cuda non supporta queste operazioni per tipi double
    ma si possono implementare mediante cast in long long int
*/
__device__ void atomicDouble(double* address, double val, bool max, bool add) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        if(max && !add) 
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed)))); //prendo il più grande tra il valore nella cella e quello di sum attuale
        else if(!max && !add)
           old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmin(val, __longlong_as_double(assumed)))); //stessa cosa per il più piccolo
        else if(!max && add) //uso per fare l'addizione
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old); //se la cella viene sovrascritta da un altro thread subito dopo che ci ha scritto dentro ricontrollo per sicurezza
}

/*
    funzione per calcolare il prodotto matrice vettore (X * e1/e2) e per memorizzare i valori massimi e minimi di c1 e c2
*/
__device__ void mulMatrix(double* c1, double* c2, double* max, double* min, double* th, double* x, int* row, int idx, int idy) {
    double e[4];
    th[idx] = step * idx;
    e[0] = cos(th[idx]);
    e[1] = sin(th[idx]);
    e[2] = -sin(th[idx]);
    e[3] = cos(th[idx]);

    double sum = 0;
    sum += x[idy] * e[0];
    sum += x[idy + row[0]] * e[1]; //il dato (idy,1) non si trova di fianco ma ad una distanza row[0] da (idy,0)
    c1[(row[0] * idx) + idy] = sum;
    //c1_max
    if (max[idx * 2] < sum) {
        atomicDouble(&max[idx * 2], sum, true, false);
    }
    //c1_min
    if (min[idx * 2] > sum) {
        atomicDouble(&min[idx * 2], sum, false, false);
    }

    sum = 0;
    sum += x[idy] * e[2];
    sum += x[idy + row[0]] * e[3];
    c2[(row[0] * idx) + idy] = sum;

    //c2_max
    if (max[idx * 2 + 1] < sum) {
        atomicDouble(&max[idx * 2 + 1], sum, true, false);
    }
    //c2_min
    if (min[idx * 2 + 1] > sum) {
        atomicDouble(&min[idx * 2 + 1], sum, false, false);
    }

}
//da riguardare gli if
/*/if (c1maxdata < c1mindata and c2maxdata < c2mindata) {
    mi = fmin(c1max[(row[0] * idx) + idy], c2max[(row[0] * idx) + idy]);
    ma = fmax(mi, d0);
    div = 1 / ma;
    atomicDouble(&b[idx], div, false, true);

}
else if (c1maxdata < c1mindata and c2maxdata >= c2mindata) {
    mi = fmin(c1max[(row[0] * idx) + idy], c2min[(row[0] * idx) + idy]);
    ma = fmax(mi, d0);
    div = 1 / ma;
    atomicDouble(&b[idx], div, false, true);

}
else if (c1maxdata >= c1mindata and c2maxdata < c2mindata) {
    mi = fmin(c1min[(row[0] * idx) + idy], c2max[(row[0] * idx) + idy]);
    ma = fmax(mi, d0);
    div = 1 / ma;
    atomicDouble(&b[idx], div, false, true);
}
else if (c1maxdata >= c1mindata and c2maxdata >= c2mindata) {
    mi = fmin(c1min[(row[0] * idx) + idy], c2min[(row[0] * idx) + idy]);
    ma = fmax(mi, d0);
    div = 1 / ma;
    atomicDouble(&b[idx], div, false, true);

}*/
__global__ void calcKernel(double* th, double* x, double* c1, double* c2, int* row, double* max, double* min, double* c1max, double* c1min, double* c2max, double* c2min, double* b) {
    int idx = blockIdx.x;
    int idy = threadIdx.x;
    max[idx * 2] = -10000;
    max[idx * 2 + 1] = -10000;
    min[idx * 2] = 10000;
    min[idx * 2 + 1] = 10000;
    __syncthreads();

    mulMatrix(c1, c2, max, min, th, x, row, idx, idy);
    __syncthreads();

    //differenza tra estremi e array C1 e C2
    c1max[(row[0] * idx) + idy] = max[idx * 2] - c1[(row[0] * idx) + idy];
    c1min[(row[0] * idx) + idy] = c1[(row[0] * idx) + idy] - min[idx * 2];
    c2max[(row[0] * idx) + idy] = max[idx * 2 + 1] - c2[(row[0] * idx) + idy];
    c2min[(row[0] * idx) + idy] = c2[(row[0] * idx) + idy] - min[idx * 2 + 1];

    __syncthreads();

    //calcolo norma
    __shared__ double c1maxdata, c1mindata, c2maxdata, c2mindata; //la shared vale per i thread di 1 blocco tenere blocchi indipendenti
    c1maxdata = 0;
    c1mindata = 0;
    c2maxdata = 0;
    c2mindata = 0;
    b[idx] = 0;

    atomicDouble(&c1maxdata, c1max[(row[0] * idx) + idy] * c1max[(row[0] * idx) + idy], false, true);
    atomicDouble(&c1mindata, c1min[(row[0] * idx) + idy] * c1min[(row[0] * idx) + idy], false, true);
    atomicDouble(&c2maxdata, c2max[(row[0] * idx) + idy] * c2max[(row[0] * idx) + idy], false, true);
    atomicDouble(&c2mindata, c2min[(row[0] * idx) + idy] * c2min[(row[0] * idx) + idy], false, true);
    __syncthreads();

    double ma, mi, div, d0 = 0.001, val1, val2;

    //usare nvprof con versione precedente per valutare i cambiamenti
    bool cond1 = c1maxdata >= c1mindata;
    bool cond2 = c2maxdata >= c2mindata;


    //CONTROLLARE SE IL CALCOLO è GIUSTO
    val1 = cond1 * c1min[(row[0] * idx) + idy] + !cond1 * c1max[(row[0] * idx) + idy]; //se la norma di c1max supera quella di c1min prendo il secondo array e viceversa
    val2 = cond2 * c2min[(row[0] * idx) + idy] + !cond2 * c2max[(row[0] * idx) + idy];
    //convertire cond1 in intero -> cond1 * c1min... + !cond1 * c1max...
    /*val1 = cond1 ? c1min[(row[0] * idx) + idy] : c1max[(row[0] * idx) + idy];
    val2 = cond2 ? c2min[(row[0] * idx) + idy] : c2max[(row[0] * idx) + idy];*/

    mi = fmin(val1, val2);
    ma = fmax(mi, d0);
    div = 1 / ma;
    atomicDouble(&b[idx], div, false, true);


    __syncthreads();

    //riutilizzo l'array max per memorizzare la matrice Q
    max[idx * 2] = th[idx];
    max[idx * 2 + 1] = b[idx];

    /*c1max[idx] = b[idx];
    __syncthreads();
    b[idx] = idx;
    __syncthreads();


    if (idy == 0)
    printf(" %i %i", idx, (int) b[idx]);
    if (idy == 0 and idx == 0) printf("\n");

    if (idx + 1 < d and idx % 2 == 0) {
        b[idx] = c1max[idx] < c1max[idx + 1] ? (double)b[idx + 1] : (double)b[idx];
        c1max[idx] = c1max[idx] < c1max[idx + 1] ? c1max[idx + 1] : c1max[idx];
    }
    __syncthreads();


    for (int s = 2; s <= d/2; s += 2) {
        if (idx + s < d and idx % 2 == 0) {
            b[idx] = c1max[idx] < c1max[idx + s] ? (double)b[idx + s] : (double)b[idx];
            c1max[idx] = c1max[idx] < c1max[idx + s] ? c1max[idx + s] : c1max[idx];
        }
        __syncthreads();
    }
    __syncthreads();

    maxId[0] = (int)b[0];*/
    
}

/*
    kernel che calcola i valori dei coefficienti
*/
__global__ void calcCoefKernel(double* c1, double* c2, double* x, double* max, double* min, int* row, double* th, double* coef, int* maxth_id) {
    int idy = threadIdx.y;
    int idx = (maxth_id[0]/2)-1; //risalgo all'indice del th che mi serve 

    max[idx * 2] = -1000;
    max[idx * 2 + 1] = -1000;
    min[idx * 2] = 10000;
    min[idx * 2 + 1] = 10000;
    __syncthreads();

    mulMatrix(c1, c2, max, min, th, x, row, idx, idy);
    __syncthreads();

    coef[0] = cos(th[idx]);
    coef[1] = sin(th[idx]);
    coef[2] = min[idx*2];
    coef[3] = -sin(th[idx]);
    coef[4] = coef[0];
    coef[5] = min[idx * 2 + 1];
    coef[6] = coef[0];
    coef[7] = coef[1];
    coef[8] = max[idx * 2];
    coef[9] = coef[3];
    coef[10] = coef[0];
    coef[11] = max[idx * 2 + 1];

       
}

double* launchKernelCuda(const double* X,const int row) {
    //alla prima iterazione il tempo impiegato è il più alto
    cudaMemcpyAsync(d_row, &row, sizeof(int), cudaMemcpyHostToDevice, s1);
    cudaMemcpyAsync(d_x, X, sizeof(double) * row * 2, cudaMemcpyHostToDevice, s1);

    calcKernel << <d, row, 0, s1 >> > (
        d_th, d_x, d_c1, d_c2,
        d_row, d_max, d_min, d_c1max,
        d_c1min, d_c2max, d_c2min, d_b
    );
    
    cublasIdamax(handle, d * 2, d_max, 1, d_idmaxth); //restituisce l'indice a partire da 1

    calcCoefKernel<<<1, dim3(1, row), 0, s1 >> >(d_c1, d_c2, d_x, d_max, d_min, d_row, d_th, d_coef, d_idmaxth);

    cudaMemcpyAsync(h_coef, d_coef, sizeof(double) * 12, cudaMemcpyDeviceToHost, s1);


    cudaStreamSynchronize(s1);
    return h_coef;

}