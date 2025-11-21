function H = MutualInformation(x, y)

%Mutual information quantifies the reduction of uncer- tainty regarding X by observing Y

N = length(x);

% remap variable values
val = unique(x);
kx = length(val);
dummy(val) = 0:kx-1;
x = dummy(x);

val = unique(y);
ky = length(val);
dummy(val) = 0:ky-1;
y = dummy(y);

Val = min([length(x), length(y)]);
x = x(1:Val);
y = y(1:Val);

% collect statistics
py = histcounts(y, 0:ky)/N;
%Hy = -sum(log2(py).* py)

px = histcounts(x, 0:kx)/N;
%Hx = -sum(log2(px).* px)

py = kron(ones(1,  kx),py);
px = kron(ones(1,  ky),px);
xy = x*ky+y;
pxy = histcounts(xy, (0:kx*ky))/N;

idx = find(pxy);
H = -sum(log2(pxy(idx)./(py(idx).*px(idx))).* pxy(idx));

%Hxy = -sum(log2(pxy(idx)).* pxy(idx))

