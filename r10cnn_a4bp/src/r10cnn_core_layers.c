#include "r10_cnn.h"
#include <stdarg.h>

void r10_idx2sub(const size_t idx, size_t * sub, const size_t * shape, const size_t ndim);
void _matmul_float(float * C, const float * A, const float * B, const size_t outrows,
                const size_t outcols, const size_t innerdim);
size_t r10_sub2idx(const size_t * sub, const size_t * shape, const size_t ndim);
void r10_dot(r10_tensor* C, const r10_tensor* A, const r10_tensor* B, const size_t * axesA,
    const size_t * axesB, const size_t naxes, const int normalize, r10_tensor* workspace);
void r10_affine_matmul(r10_tensor* ofm, r10_tensor* ifm, r10_tensor* kernel, r10_tensor* bias,
    const size_t outrows,const size_t outcols, const size_t innerdim);
void r10_softmax_func(r10_tensor* r10tensor, const size_t size);

extern TickType_t begin, elapse;

void r10_add(r10_tensor* output,...) {

    size_t num_tensors = 2; 

    va_list args;
    const r10_tensor *arrptr;
    va_start (args, num_tensors);
    memset(output->dataf, 0, output->num_data*sizeof(output->dataf[0]));

    for (size_t i = 0; i < num_tensors; ++i) {
        arrptr = va_arg(args, r10_tensor*);
        for (size_t j=0; j<output->num_data; ++j) {
            output->dataf[j] += arrptr->dataf[j];
        }
    }
    va_end (args);
}

/**
 * Converts linear indices to subscripts in row major order.
 *
 * :param idx: linear index in row major order.
 * :param sub: array[ndim] output subscript.
 * :param shape: array[ndim] shape of array being indexed.
 * :param ndim: number of dimensions of array being indexed.
 */
void r10_idx2sub(const size_t idx, size_t * sub, const size_t * shape, const size_t ndim) {

    size_t idx2 = idx;
    for (int i=ndim-1; i>=0; --i) {
        sub[i] = idx2%shape[i];
        idx2 /= shape[i];
    }
}

/**
 * Just your basic 1d matrix multipication.
 * computes C = A*B
 * assumes A,B,C are all 1d arrays of matrices stored in row major order.
 *
 * :param C: output array.
 * :param A: input array 1.
 * :param B: input array 2.
 * :param outrows: number of rows of C and A.
 * :param outcols: number of cols of C and B.
 * :param innderdim: number of cols of A and rows of B
 */
void _matmul_float(float * C, const float * A, const float * B, const size_t outrows,
                const size_t outcols, const size_t innerdim) {

    // make sure output is empty
    // memset(C, 0, outrows*outcols*sizeof(C[0]));

    for (size_t i = 0 ; i < outrows; ++i) {
        const size_t outrowidx = i*outcols;
        const size_t inneridx = i*innerdim;
        for (size_t k = 0; k < innerdim; ++k) {
            for (size_t j = 0;  j < outcols; ++j) {
                C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j];
            }
        }
    }

    return;
}

/**
 * Converts subscripts to linear indices in row major order.
 *
 * :param sub: array[ndim] subscript to convert.
 * :param shape: array[ndim] shape of array being indexed.
 * :param ndim: number of dimensions of array being indexed.
 * :return: linear index in row major order.
 */
size_t r10_sub2idx(const size_t * sub, const size_t * shape, const size_t ndim) {

    size_t idx = 0;
    size_t temp = 0;
    for (size_t i=0; i<ndim; ++i) {
        temp = sub[i];
        for (size_t j=ndim-1; j>i; --j) {
            temp *= shape[j];
        }
        idx += temp;
    }
    return idx;
}

/**
 * Dot product (tensor contraction) between 2 tensors. C=A*B
 *
 * :param C: output tensor.
 * :param A: input tensor 1.
 * :param B: input tensor 2.
 * :param axesA: array[naxes] of axes of A being contracted.
 * :param axesB: array[naxes] of axes of B being contracted.
 * :param naxes: number of axes being contracted from each input.
 * :param normalize: (0,1) whether to L2-normalize samples along the dot product axis before taking the dot product. If set to 1, then the output of the dot product is the cosine proximity between the two samples.
 * :param fwork: array of working space, size(fwork) = size(A) + size(B)
 */
void r10_dot(r10_tensor* C, const r10_tensor* A, const r10_tensor* B, const size_t * axesA,
    const size_t * axesB, const size_t naxes, const int normalize, r10_tensor* workspace) {

    size_t permA[R10_MAX_NDIM];
    size_t permB[R10_MAX_NDIM];
    size_t prod_axesA = 1;
    size_t prod_axesB = 1;
    size_t free_axesA, free_axesB;
    size_t freeA[R10_MAX_NDIM];
    size_t freeB[R10_MAX_NDIM];
    size_t count;
    int isin;
    size_t newshpA[R10_MAX_NDIM];
    size_t newshpB[R10_MAX_NDIM];
    const size_t ndimA = A->ndim;
    const size_t ndimB = B->ndim;

    size_t Asub[R10_MAX_NDIM];
    size_t Bsub[R10_MAX_NDIM];

    float *fwork, *reshapeA, *reshapeB;
    bin16 *bwork, *reshapeA_bin16, *reshapeB_bin16;

    // find which axes are free (ie, not being summed over)
    count=0;
    for (size_t i=0; i<ndimA; ++i) {
        isin = 0;
        for (size_t j=0; j<naxes; ++j) {
            if (i==axesA[j]) {
                isin=1;
            }
        }
        if (!isin) {
            freeA[count] = i;
            ++count;
        }
    }
    count=0;
    for (size_t i=0; i<ndimB; ++i) {
        isin = 0;
        for (size_t j=0; j<naxes; ++j) {
            if (i==axesB[j]) {
                isin=1;
            }
        }
        if (!isin) {
            freeB[count] = i;
            ++count;
        }
    }

    // number of elements in inner dimension
    for (size_t i=0; i < naxes; ++i) {
        prod_axesA *= A->shape[axesA[i]];
    }
    for (size_t i=0; i < naxes; ++i) {
        prod_axesB *= B->shape[axesB[i]];
    }
    // number of elements in free dimension
    free_axesA = A->num_data/prod_axesA;
    free_axesB = B->num_data/prod_axesB;
    // find permutation of axes to get into matmul shape
    for (size_t i=0; i<ndimA-naxes; ++i) {
        permA[i] = freeA[i];
    }
    for (size_t i=ndimA-naxes, j=0; i<ndimA; ++i, ++j) {
        permA[i] = axesA[j];
    }
    for (size_t i=0; i<naxes; ++i) {
        permB[i] = axesB[i];
    }
    for (size_t i=naxes, j=0; i<ndimB; ++i, ++j) {
        permB[i] = freeB[j];
    }



    for (size_t i=0; i<ndimA; ++i) {
        newshpA[i] = A->shape[permA[i]];
    }
    for (size_t i=0; i<ndimB; ++i) {
        newshpB[i] = B->shape[permB[i]];
    }

    fwork = workspace->dataf;
    reshapeA = &fwork[0];   // temp working storage
    reshapeB = &fwork[A->num_data];
    // reshape arrays
    for (size_t i=0; i<A->num_data; ++i) {
        r10_idx2sub(i,Asub,A->shape,ndimA);
        for (size_t j=0; j<ndimA; ++j) {
            Bsub[j] = Asub[permA[j]];
        }
        size_t bidx = r10_sub2idx(Bsub,newshpA,ndimA);
        reshapeA[bidx] = A->dataf[i];
    }

    for (size_t i=0; i<B->num_data; ++i) {
        r10_idx2sub(i,Bsub,B->shape,ndimB);
        for (size_t j=0; j<ndimB; ++j) {
            Asub[j] = Bsub[permB[j]];
        }
        size_t bidx = r10_sub2idx(Asub,newshpB,ndimB);
        reshapeB[bidx] = B->dataf[i];
    }


    if (normalize) {

        float sum;
        float inorm;
        for (size_t i=0; i<free_axesA; ++i) {
            sum = 0;
            for (size_t j=0; j<prod_axesA; ++j) {
                sum += reshapeA[i*prod_axesA + j]*reshapeA[i*prod_axesA + j];
            }
            inorm = 1.0f/sqrtf(sum);
            for (size_t j=0; j<prod_axesA; ++j) {
                reshapeA[i*prod_axesA + j] *= inorm;
            }
        }
        for (size_t i=0; i<free_axesB; ++i) {
            sum = 0;
            for (size_t j=0; j<prod_axesB; ++j) {
                sum += reshapeB[i + free_axesB*j]*reshapeB[i + free_axesB*j];
            }
            inorm = 1.0f/sqrtf(sum);
            for (size_t j=0; j<prod_axesB; ++j) {
                reshapeB[i + free_axesB*j] *= inorm;
            }
        }
    }

    _matmul_float(C->dataf, reshapeA, reshapeB, free_axesA,
            free_axesB, prod_axesA);

    return;
}

/**
 * Affine matrix multiplication.
 * computes C = A*B + d, where d is a vector that is added to each
 row of A*B
 * assumes A,B,C are all 1d arrays of matrices stored in row major order
 *
 * :param C: output array.
 * :param A: input array 1.
 * :param B: input array 2.
 * :param d: input array 3.
 * :param outrows: number of rows of C and A.
 * :param outcols: number of cols of C, B and d.
 * :param innderdim: number of cols of A and rows of B
 * 
 */
void r10_affine_matmul(r10_tensor* ofm, r10_tensor* ifm, r10_tensor* kernel, r10_tensor* bias,
    const size_t outrows,const size_t outcols, const size_t innerdim) {

    // float *C = ofm->dataf;
    // const float *A = ifm->dataf;
    // const float *B = kernel->dataf;
    // const float *d = bias->dataf;
    // make sure output is empty
    float temp, ifm_f, kernel_f;

    // memset(ofm->dataf, 0, outrows*outcols*sizeof(ofm->dataf[0]));

    for (size_t i = 0 ; i < outrows; ++i) {
        const size_t outrowidx = i*outcols;
        const size_t inneridx = i*innerdim;
        for (size_t j = 0;  j < outcols; ++j) {
            for (size_t k = 0; k < innerdim; ++k) {
                ofm->dataf[outrowidx+j] += ifm->dataf[inneridx+k] * kernel->dataf[k*outcols+j];
                // printf("HERE %ld\n", innerdim);
            }
            // BECAUSE THE d - bias is not defined in our scope
            // printf("bias added: %f\n", d[j]);
            ofm->dataf[outrowidx+j] += bias->dataf[j]; 
        }
    }

    return;
}

/**
 * Soft max activation function.
 *   z[i] = exp(x[i]-max(x))
 *   y = z/sum(z)
 *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 * 
 * void r10_softmax_func(float * x, const size_t size);
 */
void r10_softmax_func(r10_tensor* r10tensor, const size_t size) {

    float xmax, sum, temp;
    bin16 xmax_bin16, sum_bin16;
    // bin32 temp = 0x0;

    xmax = r10tensor->dataf[0];
    sum = 0;
    for (size_t i=0; i < size; ++i) {
        if (r10tensor->dataf[i]>xmax) {
            xmax = r10tensor->dataf[i];
        }
    }

    for (size_t i=0; i < size; ++i) {
        r10tensor->dataf[i] = expf(r10tensor->dataf[i]-xmax);
    }

    for (size_t i=0; i < size; ++i) {
        sum += r10tensor->dataf[i];
    }

    sum = 1.0f/sum;
    for (size_t i=0; i < size; ++i) {
        r10tensor->dataf[i] = r10tensor->dataf[i]*sum;
    }
    

    return;
}
// k2c_activationType * k2c_softmax = r10_softmax_func;

void r10_dense(size_t layer_id, exe_config *config, r10_tensor* kernel, r10_tensor* bias, 
    r10_tensor* ifm, r10_tensor* ofm, r10_tensor* workspace)
{
#if defined(UART_PROFILE)
    begin = xTaskGetTickCount();
#endif

    ifm->dataf = (float*)pvPortMalloc(ifm->num_data*sizeof(float));
    nvm_to_vm(ifm->dataf, ifm->num_data, ifm->nvm_start);
    // kernel->dataf = (float*)pvPortMalloc(kernel->num_data*sizeof(float));
    // nvm_to_vm(kernel->dataf, kernel->num_data, kernel->nvm_start);

    if (ifm->ndim <=2) {
        size_t outrows;

        if (ifm->ndim>1) {
            outrows = ifm->shape[0];
        }
        else {
            outrows = 1;
        }
        const size_t outcols = kernel->shape[1];
        const size_t innerdim = kernel->shape[0];
        const size_t outsize = outrows*outcols;

        
        // ==============================================
        r10_affine_matmul(ofm,ifm,kernel,bias,
                        outrows,outcols,innerdim);
        r10_softmax_func(ofm,outsize);
        // ==============================================
        
    }
    else {
        const size_t axesA[1] = {ifm->ndim-1};
        const size_t axesB[1] = {0};
        const size_t naxes = 1;
        const int normalize = 0;

        // ==============================================
        r10_dot(ofm, ifm, kernel, axesA, axesB, naxes, normalize, workspace);
        r10_bias_add(ofm,bias);
        r10_softmax_func(ofm, ofm->num_data);
        // ==============================================
    }

    if(vm_to_nvm(ofm->dataf, ofm->num_data, ofm->nvm_start, &ofm->nvm_end) != 0){
        am_util_stdio_printf("ERROR! vm_to_nvm return non-zero\n");
    }

#if defined(UART_PROFILE)
    elapse = xTaskGetTickCount() - begin;
    am_util_stdio_printf("DENSE Layer %ld: %ld\n", layer_id, elapse);
#endif
    
    return;
}
