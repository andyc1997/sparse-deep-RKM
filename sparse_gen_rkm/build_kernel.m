function K = build_kernel(X, params)
    % Build kernel matrix when implicit feature map is used

    % expect params = {'rbf_func', sig2}
    if strcmp(params{1}, 'rbf_func') % radial basis function kernel
        D = pdist(X, 'squaredeuclidean');
        K = exp(-squareform(D)./ (2*params{2}));     
   
    % expect params = {'laplace', sig}
    elseif strcmp(params{1}, 'laplace_func')
        D = pdist(X, 'euclidean');
        K = exp(-squareform(D)./ params{2});    
    
    % invalid input
    else
        catch_msg = 'invalid input for kernel function';
        error(catch_msg);
    end
    
end

% debugged