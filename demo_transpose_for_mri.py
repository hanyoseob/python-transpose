## Reference
# 1) https://en.wikipedia.org/wiki/Inner_product_space
# 2) https://en.wikipedia.org/wiki/Transpose
# 3) https://en.wikipedia.org/wiki/Complex_conjugate

## 1) Inner Product definition
# < X, Y >	= < [x1; x2; ...; xn], [y1; y2; ...; yn] >
#         	= [x1; x2; ...; xn]' * [y1; y2; ...; yn]
#         	= SUM_(i=1)^(n) xi * yi
#         	= (x1 * y1) + (x2 * y2) + ... + (xn * yn)

## 2) Transpose definition
# If,       < A * X, Y > = < X, A^T * Y >
# then,     A^T is A's transpose

## 3) Complex conjugate definition
# (a + ib)' = a - ib;

from numpy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np

## Generate data A in R ^ ( N x M ), X in R ^ ( M x K ) and Y in R ^ ( N x K )
dataType = 'IMAG'    # dataType = [COMPLEX, REAL, IMAG]
N = 100
M = N
K = N

if dataType == 'REAL':
    X = np.random.rand(M, K)
    Y = np.random.rand(N, K)
elif dataType == 'IMAG':
    X = 1j*np.random.rand(M, K)
    Y = 1j*np.random.rand(N, K)
elif dataType == 'COMPLEX':
    X = np.random.rand(M, K) + 1j*np.random.rand(M, K)
    Y = np.random.rand(N, K) + 1j*np.random.rand(N, K)

A = lambda x: (fft2((x)))/(M*K)
AT = lambda y: (ifft2((y)))

## Calculate < A * X, Y >
# A * X
AX = A(X)

# < A * X, Y >
lhs = np.dot(AX.reshape(1, -1).conj(), Y.reshape(-1, 1))

## Calculate < X, A^T * Y >
# A^T * Y
ATY = AT(Y)

# < X, A^T * Y >
rhs = np.dot(X.reshape(1, -1).conj(), ATY.reshape(-1, 1))

## Proof that
# < A * X, Y >                  = < X, A^T * Y >
# < A * X, Y > - < X, A^T * Y > = 0

th = 1e-10

print('dim( A ) = %s ^ ( %d, %d )' % (dataType, N, M))
print('dim( X ) = %s ^ ( %d, %d )' % (dataType, M, K))
print('dim( Y ) = %s ^ ( %d, %d )' % (dataType, N, K))
print(' ')

print(' < A * X, Y >                  = %.6f + %.6fj' % (lhs.real, lhs.imag))
print(' < X, A^T * Y >                = %.6f + %.6fj' % (rhs.real, rhs.imag))
print(' < A * X, Y > - < X, A^T * Y > = %.6f + %.6fj' % ((lhs - rhs).real, (lhs - rhs).imag))
print(' ')

if abs(lhs - rhs) < th:
    print('Since < A * X, Y > - < X, A^T * Y > = 0, A^T is the Transpose of A.')
else:
    print('Since < A * X, Y > - < X, A^T * Y > != 0, A^T is not the Transpose of A.')

