function [V, D, K] =  sparse_pca(X, ft_map, params, s, no_sparse, on_stm)
    % X: data matrix
    % ft_map: neural network object
    % params: parameters for kernel function {'name', value}
    % s: number of pc
    % no_sparse: 1 for sparse PCA; 0 for classical PCA
    % on_stm: 1 for Stiefel manifold; 0 for Euclidean space
    
    % Compute top s pairs of sparse eigenvectors (V) and eigenvalues (D)
    % dim(X) = num of obs * num of features
    % dim(Xf) = num of obs * new num of features
   
    state = 'explicit';
    N = size(X, 1);
    
    assert(N > 0, 'N must be positive');
    assert(s <= N, 's cannot be greater than N');
    assert(no_sparse == 1 || no_sparse == 0, 'no_sparse must be either 0 or 1')
    assert(on_stm == 1 || on_stm == 0, 'on_stm must be either 0 or 1')
    
    % check explicit or implicit feature map
    if ft_map == []
        state = 'implicit';
    end
    
    % implicit feature map
    if strcmp(state, 'implicit') == 1
        K = build_kernel(X, params);
    
    % explicit feature map
    elseif strcmp(state, 'explicit') == 1
        Xf = predict(ft_map, X);
        K = Xf*Xf';
        
    % invalid input
    else
        catch_msg = 'invalid input for feature map';
        error(catch_msg);
    end
    
    % kernel centering
    P = (eye(N) - ones(N)./N);
    K = P*K*P;
    
    % classical PCA
    [V, D] = eigs(K, s);
    D = diag(D);
    if no_sparse == 1
        return
    end
    
    if on_stm == 0
        % sparse PCA - Gpower method l0 norm, Euclidean space
        gamma = 0.1*ones(1, s);  
        V = GPower(K, gamma.^2, s, 'l0', 0);
        D = diag(V'*K*V);
        
    elseif on_stm == 1
        % sparse PCA - Gpower method l0 norm, Stiefel manifold
        gamma = 0.1*ones(1, s);
        mu = (1:s).^(-1); 
        V = GPower(K, gamma.^2, s, 'l0', 1, mu);
        D = diag(V'*K*V);
    end
end