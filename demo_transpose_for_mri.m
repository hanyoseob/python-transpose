%% Reference
% 1) https://en.wikipedia.org/wiki/Inner_product_space
% 2) https://en.wikipedia.org/wiki/Transpose
% 3) https://en.wikipedia.org/wiki/Complex_conjugate

%% 1) Inner Product definition
% < X, Y >  = < [x1; x2; ...; xn], [y1; y2; ...; yn] >
%           = [x1; x2; ...; xn]' * [y1; y2; ...; yn]
%           = SUM_(i=1)^(n) xi * yi
%           = (x1 * y1) + (x2 * y2) + ... + (xn * yn)

%% 2) Transpose definition
% If,       < A * X, Y > = < X, A^T * Y >
% then,     A^T is A's transpose

%% 3) Complex conjugate definition
% (a + ib)' = a - ib;

%% Clear the Workspace
clear;
home;

%% Generate data A in R ^ ( N x M ), X in R ^ ( M x K ) and Y in R ^ ( N x K )
dataType	= 'IMAG';    % dataType = [COMPLEX, REAL, IMAG]
N           = 100;
M           = N;
K           = N;

A   = @(x) fftshift(fft2(ifftshift(x)))./numel(x);
AT  = @(y) ifftshift(ifft2(fftshift(y)));

switch dataType
    case 'REAL'
        X   = rand(M, K);
        Y   = rand(N, K);
    case 'IMAG'
        X   = 1i*rand(M, K);
        Y   = 1i*rand(N, K);
    case 'COMPLEX'
        X   = rand(M, K) + 1i*rand(M, K);
        Y   = rand(N, K) + 1i*rand(N, K);
end


%% Calculate < A * X, Y >
% A * X
AX  = A(X);

% < A * X, Y >
lhs = AX(:)'*Y(:);

%% Calculate < X, A^T * Y >
% A^T * Y
ATY = AT(Y);

% < X, A^T * Y >
rhs	= X(:)'*ATY(:);

%% Proof that
% < A * X, Y > = < X, A^T * Y >
% < A * X, Y > - < X, A^T * Y > = 0

th  = 1e-10;

disp(['dim( A )       = ' dataType ' ^ ' num2str([N, M], '( %d,	%d )')]);
disp(['dim( X )       = ' dataType ' ^ ' num2str([M, K], '( %d,	%d )')]);
disp(['dim( Y )       = ' dataType ' ^ ' num2str([N, K], '( %d,	%d )')]);
disp(' ');

disp([' < A * X,    Y >                 = ' num2str(lhs, '%.6f') ]);
disp([' < X,        A^T * Y >           = ' num2str(rhs, '%.6f') ]);
disp([' < A * X, Y > - < X, A^T * Y >	 = ' num2str(lhs - rhs, '%.6f') ]);
disp(' ');

if (abs(lhs - rhs) < th)
    disp(['Therefore, A^T is the Transpose of A.']);
else
    disp(['Therefore, A^T is not the Transpose of A.']);
end
