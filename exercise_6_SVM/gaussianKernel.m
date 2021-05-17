% SUPPORT VECTOR MACHINE - GAUSSIAN KERNEL

function sim = gaussianKernel(x1, x2, sigma)

% returns a gaussian kernel between x1 and x2

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

v = x1-x2;
v2 = sum(v' * v);
sigma = 2*sigma^2;

sim = exp(-(v2/sigma));

    
end