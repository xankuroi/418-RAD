#ifndef __CUDAMATRIX_H_
#define __CUDAMATRIX_H_

#include <cstdlib>


namespace CUDAMatrix {
    template<typename T, size_t M, size_t N> class CUDAMatrix;

    template<typename T, size_t M, size_t N>
    class ConstCUDAMatrixRow {
        private:
            size_t m_i;
            const CUDAMatrix<T, M, N>& m_matrix;

        public:
            __device__ __host__ ConstCUDAMatrixRow(
                const CUDAMatrix<T, M, N>& matrix,
                size_t i
            );

            __device__ __host__ const T& operator[](size_t j) const;
    };

    template<typename T, size_t M, size_t N>
    class CUDAMatrixRow {
        private:
            size_t m_i;
            CUDAMatrix<T, M, N>& m_matrix;

        public:
            __device__ __host__ CUDAMatrixRow(
                CUDAMatrix<T, M, N>& matrix,
                size_t i
            );

            __device__ __host__ T& operator[](size_t j);
            __device__ __host__ const T& operator[](size_t j) const;
    };

    template<typename T, size_t M, size_t N>
    class CUDAMatrix {
        template<typename T, size_t M2, size_t N2>
        friend class CUDAMatrix;

        friend class CUDAMatrixRow<T, M, N>;
        friend class ConstCUDAMatrixRow<T, M, N>;

        private:
            T m_data[M][N];

        public:
            __device__ __host__ CUDAMatrix();
            __device__ __host__ CUDAMatrixRow<T, M, N> operator[](size_t i);
            const __device__ __host__ ConstCUDAMatrixRow<T, M, N>
                operator[](size_t i) const;

            template<size_t O>
            __device__ __host__ CUDAMatrix<T, M, O> operator*(
                const CUDAMatrix<T, N, O>& other
            ) const;
    };


    /**********************
     * ConstCUDAMatrixRow *
     **********************/

    template<typename T, size_t M, size_t N>
    __device__ __host__
    ConstCUDAMatrixRow<T, M, N>::ConstCUDAMatrixRow<T, M, N>(
            const CUDAMatrix<T, M, N>& matrix,
            size_t i
            ) :
        m_matrix(matrix),
        m_i(i) {}

    template<typename T, size_t M, size_t N>
    const __device__ __host__ T&
    ConstCUDAMatrixRow<T, M, N>::operator[](size_t j) const {
        return m_matrix.m_data[m_i][j];
    }


    /*****************
     * CUDAMatrixRow *
     *****************/

    template<typename T, size_t M, size_t N>
    __device__ __host__
    CUDAMatrixRow<T, M, N>::CUDAMatrixRow<T, M, N>(
            CUDAMatrix<T, M, N>& matrix,
            size_t i
            ) :
        m_matrix(matrix),
        m_i(i) {}

    template<typename T, size_t M, size_t N>
    __device__ __host__ T&
    CUDAMatrixRow<T, M, N>::operator[](size_t j) {
        return m_matrix.m_data[m_i][j];
    }

    template<typename T, size_t M, size_t N>
    const __device__ __host__ T&
    CUDAMatrixRow<T, M, N>::operator[](size_t j) const {
        return m_matrix.m_data[m_i][j];
    }


    /**************
     * CUDAMatrix *
     **************/

    template<typename T, size_t M, size_t N>
    __device__ __host__ CUDAMatrix<T, M, N>::CUDAMatrix() {};

    template<typename T, size_t M, size_t N>
    __device__ __host__ CUDAMatrixRow<T, M, N>
    CUDAMatrix<T, M, N>::operator[](size_t i) {
        CUDAMatrix<T, M, N>& self = *this;
        return CUDAMatrixRow<T, M, N>(self, i);
    }

    template<typename T, size_t M, size_t N>
    const __device__ __host__ ConstCUDAMatrixRow<T, M, N>
    CUDAMatrix<T, M, N>::operator[](size_t i) const {
        const CUDAMatrix<T, M, N>& self = *this;
        return ConstCUDAMatrixRow<T, M, N>(self, i);
    }

    template<typename T, size_t M, size_t N>
    template<size_t P>
    __device__ __host__ CUDAMatrix<T, M, P>
    CUDAMatrix<T, M, N>::operator*(
            const CUDAMatrix<T, N, P>& other
            ) const {
            
        CUDAMatrix<T, M, P> result;

        for (size_t i=0; i<M; i++) {
            for (size_t j=0; j<P; j++) {
                T sum = static_cast<T>(0);
                for (size_t k=0; k<N; k++) {
                    sum += (*this)[i][k] * other[k][j];
                }

                result[i][j] = sum;
            }
        }

        return result;
    }

}

#endif
