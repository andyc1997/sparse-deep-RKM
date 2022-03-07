cd 'C:\Users\user\Documents\Statistics\SPCA-DRKM-2021JUL01\work\new-function\shallow_kernel_machine\sparse_gen_rkm'

%% Read dataset
img = loadMNISTImages('..\train-images-idx3-ubyte\train-images.idx3-ubyte');


%% Sample data without replacement
N_subset = 5000;
rng(80116);
Xtr = datasample(img, N_subset, 2, 'Replace', false);


%% Train
s = 10;
N = N_subset;
K = build_kernel(Xtr', {'rbf_func', 0.5});
P = (eye(N) - ones(N)./N);
K = P*K*P;

[V, D] = eigs(K, s);
%%
gamma = 0.1*ones(1, s);  
Vs = GPower(K, gamma.^2, s, 'l0', 0);

%%
% src: https://github.com/davidstutz/matlab-mnist-two-layer-perceptron/blob/master/loadMNISTImages.m
function images = loadMNISTImages(filename)

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end