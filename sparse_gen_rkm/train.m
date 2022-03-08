clc; clear;
cd 'C:\Users\user\Documents\Statistics\SPCA-DRKM-2021JUL01\work\new-function\shallow_kernel_machine\sparse_gen_rkm'

%% Artificial dataset with circle
rng(80116)
r = 0.5;
th = 0:pi/100:2*pi;
x = r * cos(th);
y = r * sin(th);

x = x + 0.2*randn(size(x));
y = y + 0.2*randn(size(y));
Xtr = [x', y'];

%% Read dataset
img = loadMNISTImages('..\train-images-idx3-ubyte\train-images.idx3-ubyte');


%% Sample data without replacement
N_subset = 3000;
N_valid = 750;
seed = 80116;

rng(seed);
idx_tr = datasample(1:size(img, 2), N_subset, 'Replace', false);
idx_ts = datasample(setdiff(1:size(img, 2), idx_tr), N_valid, 'Replace', false);

% Xtr = img(:, idx_tr)'; % train set
% Xtr = Xtr + 0.2*randn(size(Xtr));
% Xts = img(:, idx_ts)'; % test set

%% Train with implicit kernel
s = 2;
params = {'rbf_func', sqrt(0.1), 'eta', 1};
[V, D, K] = sparse_kernel_pca_rkm(Xtr, [], params, s, 1, 0); % classical kPCA
[Vs, Ds, ~] = sparse_kernel_pca_rkm(Xtr, [], params, s, 0, 0); % sparse kPCA
[Vst, Dst, ~] = sparse_kernel_pca_rkm(Xtr, [], params, s, 0, 1); % sparse kPCA on St manifold

%% Gen latent distribution
Hgen = gen_latent(V, 1, 300);
Xgen = gen_new_x(Xtr, K, params, V, Hgen, 100);

%%
plot(x, y, '.b', Xgen(:, 1), Xgen(:, 2), '.r')

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
% Convert to double
images = double(images) / 255.0;

end