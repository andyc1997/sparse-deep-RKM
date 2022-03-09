clc; clear;
cd 'C:\Users\user\Documents\Statistics\SPCA-DRKM-2021JUL01\work\new-function\sparse_deep_rkm\sparse_gen_rkm'

%% Artificial dataset with circle
% rng(80116)
% r = 0.5;
% th = 0:pi/100:2*pi;
% x = r * cos(th);
% y = r * sin(th);
% 
% x = x + 0.2*randn(size(x));
% y = y + 0.2*randn(size(y));
% Xtr = [x', y'];

%% Read dataset
img = loadMNISTImages('..\mnist\train-images.idx3-ubyte');


%% Sample data without replacement
N_subset = 3000;
N_valid = 350;
seed = 80116;

rng(seed);
idx_tr = datasample(1:size(img, 2), N_subset, 'Replace', false);
idx_ts = datasample(setdiff(1:size(img, 2), idx_tr), N_valid, 'Replace', false);

Xtr = img(:, idx_tr)'; % train set
Xtr = Xtr + 0.4*randn(size(Xtr));
Xts = img(:, idx_ts)'; % test set
Xts = Xts + 0.4*randn(size(Xts));


%% Train with implicit kernel
s = 200;
params = {'rbf_func', 5^2, 'eta', 1, 'gamma', 0.001};

[V, D, K] = sparse_kernel_pca_rkm(Xtr, [], params, s, 1, 0); % classical kPCA
[Vs, Ds, ~] = sparse_kernel_pca_rkm(Xtr, [], params, s, 0, 0); % sparse kPCA
%[Vst, Dst, ~] = sparse_kernel_pca_rkm(Xtr, [], params, s, 0, 1); % sparse kPCA on St manifold

%% Gen new features
Hgen = gen_latent(Vs, 1, 10);
Xgen = gen_new_x(Xtr, K, params, Vs, Hgen, 20);

for i = 1:10
    subplot(1, 10, i)
    imshow(reshape(Xgen(i, :), 28, 28));
end

%% Preimage problem by fixed point iteration
mean_sq_err = zeros(size(Xts, 1), 1);
X_dn = zeros(size(Xts));
s_select = 50;

for i = 1:100
    [Xpre, converge, counter] = preimg_sparse_kernel_pca_rkm(Xts(i, :), Xtr, Vs(:, 1:s_select)*Vs(:, 1:s_select)', K, params);
    mean_sq_err(i) = mse(Xpre - Xts(i, :));
    X_dn(i, :) = Xpre;
    
    if mod(i, 10) == 0
        disp(['Iteration = ', num2str(i)])
    end
end

avg_err = mean(mean_sq_err);

%%
for i = 1:10
    subplot(2, 10, i)
    imshow(reshape(Xts(i, :), 28, 28));
    subplot(2, 10, i+10)
    imshow(reshape(X_dn(i, :), 28, 28));
end

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