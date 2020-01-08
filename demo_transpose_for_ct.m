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

%% Generate data A in R ^ ( N x M ), X in R ( M x K ) and Y in R ^ ( N x K )
dataType	= 'REAL';    % dataType = [COMPLEX, REAL, IMAG]
M           = 256;
K           = 100;

VIEW        = 360;

THETA       = linspace(0, 180, VIEW + 1);
THETA(end)  = [];

DCT         = size(radon(zeros(M * (M > K) + K * (K >= M)), THETA), 1);

A           = @(x)  radon(x, THETA);
AT          = @(y)  iradon(y, THETA, 'none', M * (M > K) + K * (K >= M))/(pi/(2*length(THETA)));

X           = rand(M, K);
Y           = rand(DCT, VIEW);

if (M > K)
    X 	= padarray(X, [0, floor((M - K)/2)], 'pre');
    X 	= padarray(X, [0, M - (K + floor((M - K)/2))], 'post');
else
    X 	= padarray(X, [floor((K - M)/2), 0], 'pre');
    X 	= padarray(X, [K - (M + floor((K - M)/2)), 0], 'post');
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

th  = max(abs(lhs), abs(rhs)) .* 1e-4;

disp(['dim( A )       = ' dataType ' ^ ' num2str([DCT, VIEW, M, K], '( (%d,	%d) x (%d, %d) )')]);
disp(['dim( X )       = ' dataType ' ^ ' num2str([M, K], '( %d,	%d )')]);
disp(['dim( Y )       = ' dataType ' ^ ' num2str([DCT, VIEW], '( %d,	%d )')]);
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
