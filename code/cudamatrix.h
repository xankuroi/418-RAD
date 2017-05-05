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
                ) :
                m_matrix(matrix),
                m_i(i) {}
                
            __device__ __host__ const T& operator[](size_t j) const {
                return m_matrix.m_data[m_i][j];
            }
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
                ) :
                m_matrix(matrix),
                m_i(i) {}

            __device__ __host__ T& operator[](size_t j) {
                return m_matrix.m_data[m_i][j];
            }
            
            __device__ __host__ const T& operator[](size_t j) const {
                return m_matrix.m_data[m_i][j];
            }
    };

    template<typename T, size_t M, size_t N>
    class CUDAMatrix {
        friend class CUDAMatrixRow<T, M, N>;
        friend class ConstCUDAMatrixRow<T, M, N>;

        private:
            T m_data[M][N];

        public:
            __device__ __host__ CUDAMatrixRow<T, M, N> operator[](size_t i) {
                return CUDAMatrixRow<T, M, N>(*this, i);
            }
            
            const __device__ __host__ ConstCUDAMatrixRow<T, M, N>
            operator[](size_t i) const {
                return ConstCUDAMatrixRow<T, M, N>(*this, i);
            }

            template<size_t P>
            __device__ __host__ CUDAMatrix<T, M, P> operator*(
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
    };
}

#endif
