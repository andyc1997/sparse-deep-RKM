function [Xpre, converge, counter] = preimg_sparse_kernel_pca_rkm(Xhat, Xtr, Htr2, K, params)
    % Preimage for sparse kernel PCA with RKM
    % Using implicit feature map
    N = size(Xtr, 1);
    assert(size(Xhat, 1) == 1, 'Only 1 input each time')
    assert(N > 1, 'Xtr must contain >1 obs')
    
    % fixed point iteration
    converge = 0;
    counter = 0;
    
    % similarities between Xhat and Xtr
    beta = Htr2 * exp(-sum((Xhat - Xtr).^2, 2) / 2*params{2});
    
    Xprev = Xhat;
    while true
        
        if converge == 1
            return
        end
        
        % similarities between Xprev and Xtr
        k = exp(-sum((Xprev - Xtr).^2, 2) / 2*params{2});
        smooth = beta .* k;
        Xnext = sum(diag(smooth)* Xtr, 1) / sum(smooth);
        
        err = mse(Xnext - Xprev);
        counter = counter + 1;
        
        % convergence achieved
        if err < 1E-4 
            converge = 1;
            Xpre = Xnext;
            return
        end
        
        % iteration exceeded
        if counter > 800
            converge = 0;
            Xpre = Xnext;
            return
        end
        
        Xprev = Xnext;
    end
    
end