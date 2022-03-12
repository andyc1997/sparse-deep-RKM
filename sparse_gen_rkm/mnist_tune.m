clc; clear;

%% Read dataset
N_subset = 3000; N_valid = 750;
[Xtr, Xts, Ytr, Yts] = read_mnist(80116, N_subset, N_valid, '..\mnist');

%% one versus all encoding
Ysgn_tr = one_versus_all_encode(Ytr);
Ysgn_ts = one_versus_all_encode(Yts);


%% Tune sigma
sig_list_greedy = [10^-2, 10^-1, 10^0, 10^1, 10^2]; % greedy search over a wide range
sig_list_refine = 5 + linspace(-2, 2, 10); % search in a refined region

% search
for sig = sig_list_refine  
    error = 0;
    
    % multiclass LSSVM
    params = {'rbf_func', sig, 'nclass', 10, 'eta', 1, 'lambda', 1};
    [Htr, b] = cl_LSSVM(Xtr, Ysgn_tr, params);
    
    % prediction
    for i = 1:N_valid
        Ypred = pred_LSSVM(Xts(i, :), Xtr, Htr, b, params);
        error = error + any(Ypred ~= Ysgn_ts(i, :));
    end
    
    disp(['Average error rate: ', num2str(error/N_valid), ' sig: ', num2str(sig)])
end

