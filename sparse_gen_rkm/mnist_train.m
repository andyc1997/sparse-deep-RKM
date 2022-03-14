clc; clear;

% read dataset
N_subset = 3000; N_valid = 750;
[Xtr, ~, Ytr, ~] = read_mnist(80116, N_subset, N_valid, '..\mnist');

% one-hot encoding
Ytr = dummyvar(Ytr + 1);

% train RKM
% set param
s = 200;
params = {'rbf_func', 5.2222, 'eta', 1, ...
    'gamma', [0.1, 0.07, 0.07, 0.0001*ones(1,s-3)]};

% classical RKM
[V, ~, K] =  singleview_skpca_rkm(Xtr,  [], params, s, 'PCA');

% sparse RKM
[Vs, ~, ~] =  singleview_skpca_rkm(Xtr, [], params, s, 'GPowerl0');


%%
% generate hidden representation
l = 10;
Ngen = 10;
[~, h_dist] = gen_latent(Vs, l, Ngen); % V or Vs

%% synthetic data
Hgen = random(h_dist, Ngen);

Nr = 10; % number of neighborhood points
gen_params = {'rbf_func', params{2}(1), 'eta', params{4}};
Xgen = gen_new_x(Xtr, K, gen_params, Vs, Hgen, Nr);

%% visualization
for i = 1:Ngen
    subplot(1, Ngen, i)
    imshow(reshape(Xgen(i, :), 28, 28))
end

%% Traversal along eigenvector
tra_dim = 3; % dimension of traversal
another_dim = 1;
tra_size = 10; % how image for traversal

Vtra = Vs; % V or Vs

% Min and max for eigenvectors
min_val_1 = min(Vtra(:, tra_dim));
max_val_1 = max(Vtra(:, tra_dim));

min_val_2 = min(Vtra(:, another_dim));
max_val_2 = max(Vtra(:, another_dim));

id = 3; % id of Hgen

% traversal points
v1 = linspace(min_val_1, max_val_1, tra_size);
v2 = linspace(min_val_2, max_val_2, tra_size);

for k = 1:10
    % Latent space for traversal, keeping other dimension fixed
    Htra = repmat(Hgen(id, 1:end), tra_size, 1);
    Htra(:, tra_dim) = v1';
    Htra(:, another_dim) = v2(k);
    
    % Generate image Xtra from latent space Htra
    gen_params = {'rbf_func', params{2}(1), 'eta', params{4}};
    Xtra = gen_new_x(Xtr, K, gen_params, Vtra, Htra, 10);

    % visualization
    for i = 1:tra_size
        subplot(10, tra_size, i + 10*(k - 1))
        imshow(reshape(Xtra(i, :), 28, 28))
    end
end

%% Bilinear interpolation
% H1 = Vs(10, :);
% H2 = Vs(95, :);
% H3 = Vs(50, :);
% H4 = Vs(27, :);
% 
% scl = linspace(0, 1, 10);
% scl2 = linspace(0, 1, 10);
% 
% for i = 1:length(scl)
%     for j = 1:length(scl2)
%         Hintp = (1 - scl(i))*(1 - scl2(j))*H1 + scl(i)*(1 - scl2(j))*H2 + ...
%             (1 - scl(i))*scl2(j)*H3 + scl(i)*scl2(j)*H4;
%         Xintp = gen_new_x(Xtr, K, gen_params, Vs, Hintp, Nr);
%         
%         subplot(length(scl), length(scl2), (i - 1)*10 + j)
%         imshow(reshape(Xintp, 28, 28))
%     end
% end







