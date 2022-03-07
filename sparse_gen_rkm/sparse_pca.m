function [V, D, K] =  sparse_pca(X, ft_map, params, s)
    % Compute top s pairs of sparse eigenvectors (V) and eigenvalues (D)
    % dim(X) = number of obs * number of features
    
    state = 'explicit';
    N = size(X, 1);
    K = zeros(N);
    
    assert(N > 0, 'number of sample must be positive');
    assert(s <= N, 'number of the principal components cannot be greater than the sample size');
    
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
        K = Xf'*Xf;
        
    % invalid input
    else
        catch_msg = 'invalid input for feature map';
        error(catch_msg);
    end
    
    % kernel centering
    P = (eye(N) - ones(N)./N);
    K = P*K*P;
    
    % sparse PCA
    
end