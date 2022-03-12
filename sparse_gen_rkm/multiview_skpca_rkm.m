function [V, D, K] =  multiview_skpca_rkm(X, Y, ft_map_x, ft_map_y, params, s, method)
    % ****************************************************************
    % X: data matrix, view1
    % Y: data matrix, view2
    % ft_map_x / ft_map_y: neural network object
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
    % params = {'rbf_func', array of 1 by 2, 'eta1', double, 'eta2', double}
    %
    % GPowerl0:
    % params = {'rbf_func', array of 1 by 2, 'eta1', double, 'eta2', double, 'gamma', double or array of size 1 by s}
    %
    % GPowerl0_block:
    % params = {'rbf_func', array of 1 by 2, 'eta1', double, 'eta2', double, 'gamma', double or array of size 1 by s,
    % 'mu', [] or array of size 1 by s}
    %
    % MMl0:
    % params = {'rbf_func', array of 1 by 2, 'eta1', double, 'eta2', double, 'rhos', double or array of size 1 by s}
    %
    % TPower:
    % params = {'rbf_func', array of 1 by 2, 'eta1', double, 'eta2', double, 'rhos', double or array of size 1 by s,
    % rf: double}
    %
    % 
    % ****************************************************************
    
    state = 'explicit';
    Nx = size(X, 1);
    Ny = size(Y, 1);
    
    assert((Nx > 0 && Ny > 0) && (Nx == Ny), 'Nx and Ny must be positive, and equal');
    
    N = Nx;
    assert(s <= N, 's cannot be greater than N');
    
    assert(strcmp(params{3}, 'eta1') && strcmp(params{5}, 'eta2'), 'eta1 and eta2 must be provided');
    assert((params{4} > 0) && (params{6} > 0), 'eta1 and eta2 must be positive scalars');
    
    
    % check explicit or implicit feature map
    if isempty(ft_map_x) && isempty(ft_map_y) == 1
        state = 'implicit';
    end
    
    % implicit feature map
    if strcmp(state, 'implicit') == 1
        Kparams = {'rbf_func', params{2}(1)};
        Kx = build_kernel(X, Kparams);
        Kparams = {'rbf_func', params{2}(2)};
        Ky = build_kernel(Y, Kparams);
    
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
    P = eye(N) - ones(N)/N;
    K = P*(Kx/params{4} + Ky/params{6})*P;
    
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
            assert(strcmp(params{7}, 'gamma'), 'sparsity factor gamma must be provided');
            gamma = params{8};
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
            assert(strcmp(params{7}, 'gamma'), 'sparsity factor gamma must be provided');
            gamma = params{8};
            [~, ngam] = size(gamma);
            if ngam == 1
                gamma = gamma*ones(1, s);
            end
            
            % check if mu is passed as an argument
            assert(strcmp(params{9}, 'mu'), 'parameter mu must be provided');
            if isempty(params{10}) == 1
                mu = (1:s).^(-1);
            else
                mu = params{10};
                [~, nmu] = size(mu);
                assert(nmu == s, 'mu must be of length s');
            end
            
            % run sparse pca
            V = GPower(Xg, gamma, s, 'l0', 1, mu);
            D = diag(V'*K*V);
       
        case 'MMl0' % MM algorithm with l0 norm
            % check if rhos are passed as an argument
            assert(strcmp(params{7}, 'rhos'), 'penalty coefficient rhos must be provided');
            rhos = params{8};
            [~, nrhos] = size(rhos);
            if nrhos == 1
                rhos = rhos*ones(1, s);
            end
            
            % run sparse pca
            V = mm_sparse_eigen(K, [], s, rhos);
            D = diag(V'*K*V);
        
        case 'TPower' % TPower method
            % check if rhos are passed as an argument
            assert(strcmp(params{7}, 'rhos'), 'cardinality rhos must be provided');
            rhos = params{8};
            [~, nrhos] = size(rhos);
            if nrhos == 1
                rhos = rhos*ones(1, s);
            end
            
            % check if rf is provided
            assert(strcmp(params{9}, 'rf'), 'reducing factor must be provided');
            rf = params{10};
            
            % run sparse pca
            V = TPower(K, s, rhos, rf);
            D = diag(V'*K*V);
            
        otherwise
            catch_msg = 'invalid method for sparse PCA';
            error(catch_msg);
    end
end