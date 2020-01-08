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

from skimage.transform import radon, iradon
import numpy as np

## Generate data A in R ^ ( N x M ), X in R ( M x K ) and Y in R ^ ( N x K )
dataType = 'REAL'
M = 256
K = 100
ANG = 180
VIEW = 360
THETA = np.linspace(0, ANG, VIEW, endpoint=False)

A = lambda x: radon(x, THETA, circle=False).astype(np.float32)
AT = lambda y: iradon(y, THETA, circle=False, filter=None, output_size=M * (M > K) + K * (K >= M)).astype(np.float32)/(np.pi/(2*len(THETA)))
AINV = lambda y: iradon(y, THETA, circle=False, output_size=[M * (M > K) + K * (K >= M)]).astype(np.float32)

DCT = A(np.zeros((M * (M > K) + K * (K >= M),  M * (M > K) + K * (K >= M)))).shape[0]

##
X = np.random.rand(M, K)
Y = np.random.rand(DCT, VIEW)

if M > K:
    X = np.pad(X, ((0, 0), (((M - K)//2),  M - (K + ((M - K)//2)))),constant_values=0)
else:
    X = np.pad(X, ((((M - K) // 2), M - (K + ((M - K) // 2))), (0, 0)), constant_values=0)

## Calculate < A * X, Y >
# A * X
AX = A(X)

# < A * X, Y >
lhs = np.dot(AX.reshape(1, -1).conj(), Y.reshape(-1, 1))

## Calculate < X, A^T * Y >
# A^T * Y
ATY = AT(Y)

if M > K:
    ATY[:, :((M - K)//2)] = 0
    ATY[:, -(M - (K + ((M - K)//2))):] = 0
else:
    ATY[:((M - K) // 2), :] = 0
    ATY[-(M - (K + ((M - K) // 2))):, :] = 0

# < X, A^T * Y >
rhs = np.dot(X.reshape(1, -1).conj(), ATY.reshape(-1, 1))

## Proof that
# < A * X, Y >                  = < X, A^T * Y >
# < A * X, Y > - < X, A^T * Y > = 0

th = 1e-4

print('dim( A ) = %s ^ ( %d x %d, %d x %d )' % (dataType, DCT, VIEW, M, K))
print('dim( X ) = %s ^ ( %d x %d )' % (dataType, M, K))
print('dim( Y ) = %s ^ ( %d x %d )' % (dataType, DCT, VIEW))
print(' ')

print(' < A * X, Y >                  = %.6f + %.6fj' % (lhs.real, lhs.imag))
print(' < X, A^T * Y >                = %.6f + %.6fj' % (rhs.real, rhs.imag))
print(' < A * X, Y > - < X, A^T * Y > = %.6f + %.6fj' % ((lhs - rhs).real, (lhs - rhs).imag))
print(' ')

if abs(lhs - rhs) < max(abs(lhs), abs(rhs))*th:
    print('Since < A * X, Y > - < X, A^T * Y > = 0, A^T is the Transpose of A.')
else:
    print('Since < A * X, Y > - < X, A^T * Y > != 0, A^T is not the Transpose of A.')

