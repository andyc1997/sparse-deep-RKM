function [V, D, K] =  multiview_skpca_rkm(X, Y, ft_map_x, ft_map_y, params, s, no_sparse, on_stm)
    % ****************************************************************
    % X: data matrix, view1
    % Y: data matrix, view2
    % ft_map: neural network object
    % params: parameters for kernel function {'name', value}
    % s: number of pc
    % no_sparse: 0 for sparse PCA; 1 for classical PCA
    % on_stm: 0 for Euclidean space; 1 for Stiefel manifold
    
    % Compute top s pairs of sparse eigenvectors (V) and eigenvalues (D)
    % dim(X) = num of obs * num of features
    % dim(Xf) = num of obs * new num of features
    % ****************************************************************
    
    state = 'explicit';
    Nx = size(X, 1);
    Ny = size(Y, 1);
    
    assert((Nx > 0 && Ny > 0) && (Nx == Ny), 'Nx and Ny must be positive, and equal');
    assert(s <= Nx, 's cannot be greater than Nx + Ny');
    assert(no_sparse == 1 || no_sparse == 0, 'no_sparse must be either 0 or 1')
    assert(on_stm == 1 || on_stm == 0, 'on_stm must be either 0 or 1')
    assert(strcmp(params{3}, 'eta1') && strcmp(params{5}, 'eta2'), 'eta1 and eta2 must be provided')
    
    if no_sparse == 0
        assert(strcmp(params{7}, 'gamma'), 'sparsity factor must be provided')
    end
    
    N = Nx;
    
    % check explicit or implicit feature map
    if isempty(ft_map_x) && isempty(ft_map_y) == 1
        state = 'implicit';
    end
    
    % implicit feature map
    if strcmp(state, 'implicit') == 1
        Kx = build_kernel(X, params);
        Ky = build_kernel(Y, params);
    
    % explicit feature map
    elseif strcmp(state, 'explicit') == 1
        Xf_x = predict(ft_map_x, X);
        Kx = Xf_x*Xf_x';
        Xf_y = predict(ft_map_y, Y);
        Ky = Xf_y*Xf_y';
        
    % invalid input
    else
        catch_msg = 'invalid input for feature map';
        error(catch_msg);
    end
    
    % kernel centering
    P = eye(N) - ones(N)./N;
    K = P*(1/params{4} * Kx + 1/params{6} * Ky)*P / params{4};
    
    % classical PCA
    [V, D] = eigs(K, s);
    D = diag(D);
    if no_sparse == 1
        return
    end
    
    if on_stm == 0
        % sparse PCA - Gpower method l0 norm, Euclidean space
        gamma = params{8}*ones(1, s);  
        R = chol(K + diag(1E-10*ones(1, N)));
        V = GPower(R', gamma.^2, s, 'l0', 0);
        D = diag(V'*K*V);
        
    elseif on_stm == 1
        % sparse PCA - Gpower method l0 norm, Stiefel manifold
        gamma = params{8}*ones(1, s);
        mu = (1:s).^(-1); 
        R = chol(K + diag(1E-10*ones(1, N)));
        V = GPower(R', gamma.^2, s, 'l0', 1, mu);
        D = diag(V'*K*V);
    end
end