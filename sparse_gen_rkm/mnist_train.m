clc; clear;

%% read dataset
N_subset = 3000; N_valid = 750;
[Xtr, ~, Ytr, ~] = read_mnist(80116, N_subset, N_valid, '..\mnist');

% one-hot encoding
Ytr = dummyvar(Ytr + 1);

%% train RKM
% set param
s = 200;
params = {'rbf_func', 5.2222*ones(1, 2), 'eta1', 1, 'eta2', 1, ...
    'gamma', 0.01};

% classical RKM
[V, ~, ~] =  multiview_skpca_rkm(Xtr, Ytr, [], [], params, s, 'PCA');

% sparse RKM
[Vs, ~, ~] =  multiview_skpca_rkm(Xtr, Ytr, [], [], params, s, ...
    'GPowerl0');

%% generate hidden representation
l = 10;
Ngen = 10;
Hgen = gen_latent(Vs, l, Ngen);

%% compute each kernel
Kparams = {'rbf_func', params{2}(1)};
Kx = build_kernel(Xtr, Kparams);
Kparams = {'rbf_func', params{2}(2)};
Ky = build_kernel(Ytr, Kparams);

P = eye(N_subset) - ones(N_subset)/N_subset;
Kx = P*(Kx/params{4})*P;
Ky = P*(Ky/params{6})*P;

%% synthetic data
Nr = 20; % number of neighborhood points
gen_params = {'rbf_func', params{2}(1), 'eta', params{4}};
Xgen = gen_new_x(Xtr, Kx, gen_params, Vs, Hgen, Nr);
gen_params = {'rbf_func', params{2}(2), 'eta', params{6}};
Ygen = gen_new_x(Ytr, Ky, gen_params, Vs, Hgen, Nr);

%% visualization
for i = 1:Ngen
    subplot(1, Ngen, i)
    imshow(reshape(Xgen(i, :), 28, 28))
end

label = 0:9;
for i = 1:Ngen
    disp(label(logical(Ygen(i, :))))
end

