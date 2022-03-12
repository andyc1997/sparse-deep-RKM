function [V, D, K] =  singleview_skpca_rkm(X, ft_map, params, s, method)
    % ****************************************************************
    % X: data matrix, view1
    % ft_map: neural network object
    % params: parameters {'name', value}
    % s: number of pc
    % method: PCA, GPowerl0, GPowerl0_block, MMl0, TPower
    %
    % Compute top s pairs of sparse eigenvectors (V) and eigenvalues (D)
    % dim(X) = num of obs * num of features
    % dim(Xf) = num of obs * new num of features
    %
    % Syntax
    %
    % PCA:
    % params = {'rbf_func', double, 'eta', double}
    %
    % GPowerl0:
    % params = {'rbf_func', double, 'eta', double, 'gamma', double or array of size 1 by s}
    %
    % GPowerl0_block:
    % params = {'rbf_func', double, 'eta', double, 'gamma', double or array of size 1 by s,
    % 'mu', [] or array of size 1 by s}
    %
    % MMl0:
    % params = {'rbf_func', double, 'eta', double, 'rhos', double or array of size 1 by s}
    %
    % TPower:
    % params = {'rbf_func', double, 'eta', double, 'rhos', double or array of size 1 by s,
    % rf: double}
    %
    % 
    % ****************************************************************
    
    state = 'explicit';
    N = size(X, 1);
    
    assert(N > 0, 'N must be positive');
    assert(s <= N, 's cannot be greater than N');
    
    assert(strcmp(params{3}, 'eta'), 'eta must be provided');
    assert(params{4} > 0, 'eta must be a positive scalar');
    
    
    % check explicit or implicit feature map
    if isempty(ft_map) == 1
        state = 'implicit';
    end
    
    % implicit feature map
    if strcmp(state, 'implicit') == 1
        Kparams = {'rbf_func', params{2}};
        K = build_kernel(X, Kparams);
    
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
    P = eye(N) - ones(N)/N;
    K = P*(K/params{4})*P;
    
    % classical PCA
    [V, D] = eigs(K, s);
    D = diag(D);
    
    switch method
        case 'PCA' % kernal PCA
            return
        
        case 'GPowerl0' % sparse kernel PCA - Gpower method l0 norm
            % guess data matrix
            Xg = V*sqrt(abs(diag(D)))*V';        
            
            % check if gamma is a scalar, expand as an array
            assert(strcmp(params{5}, 'gamma'), 'sparsity factor gamma must be provided');
            gamma = params{6};
            [~, ngam] = size(gamma);
            if ngam == 1
                gamma = gamma*ones(1, s);
            end
            
            % run sparse pca
            V = GPower(Xg, gamma, s, 'l0', 0);
            D = diag(V'*K*V);
        
        case 'GPowerl0_block' % sparse PCA - Gpower method l0 norm, Stiefel manifold
            % guess data matrix
            Xg = V*sqrt(abs(diag(D)))*V';        
            
            % check if gamma is a scalar, expand as an array
            assert(strcmp(params{5}, 'gamma'), 'sparsity factor gamma must be provided');
            gamma = params{6};
            [~, ngam] = size(gamma);
            if ngam == 1
                gamma = gamma*ones(1, s);
            end
            
            % check if mu is passed as an argument
            assert(strcmp(params{7}, 'mu'), 'parameter mu must be provided');
            if isempty(params{8}) == 1
                mu = (1:s).^(-1);
            else
                mu = params{8};
                [~, nmu] = size(mu);
                assert(nmu == s, 'mu must be of length s');
            end
            
            % run sparse pca
            V = GPower(Xg, gamma, s, 'l0', 1, mu);
            D = diag(V'*K*V);
       
        case 'MMl0' % MM algorithm with l0 norm
            % check if rhos are passed as an argument
            assert(strcmp(params{5}, 'rhos'), 'penalty coefficient rhos must be provided');
            rhos = params{6};
            [~, nrhos] = size(rhos);
            if nrhos == 1
                rhos = rhos*ones(1, s);
            end
            
            % run sparse pca
            V = mm_sparse_eigen(K, [], s, rhos);
            D = diag(V'*K*V);
        
        case 'TPower' % TPower method
            % check if rhos are passed as an argument
            assert(strcmp(params{5}, 'rhos'), 'cardinality rhos must be provided');
            rhos = params{6};
            [~, nrhos] = size(rhos);
            if nrhos == 1
                rhos = rhos*ones(1, s);
            end
            
            % check if rf is provided
            assert(strcmp(params{7}, 'rf'), 'reducing factor must be provided');
            rf = params{8};
            
            % run sparse pca
            V = TPower(K, s, rhos, rf);
            D = diag(V'*K*V);
            
        otherwise
            catch_msg = 'invalid method for sparse PCA';
            error(catch_msg);
    end
end