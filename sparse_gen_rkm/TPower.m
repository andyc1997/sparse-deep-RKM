function V = TPower(A, s, rhos, rf)
    % ***************************************************************
    % Truncated Power Method for Sparse Eigenvalue Problems (Yuan and Zhang, 2013)
    % A: Symmetric positive semi-definite matrix
    % s: number of PCs
    % rhos: cardinality, proportion of number of obs.
    % rf: factor of reduction.
    % ***************************************************************
    [m, ~] = size(A);
    V = zeros(m, s);
    
    for comp = 1:s      
        rho = rhos(comp);
        
        % x0 sequential initialization with usual eigendecomposition
        card = m;
        
        if isempty(rf) == 1
            rf = 0.7; % default reduction factor
        end
        [x_dom, ~] = eigs(A, 1);
        
        % sequential decreasing in cardinality
        while card > floor(rho*m)
            card = floor(rf*card);
            x_dom = TPower_leading_eig(A, x_dom, card);
        end
        
        % Final update with prespecified cardinality
        x_dom = TPower_leading_eig(A, x_dom, floor(rho*m));
        V(:, comp) = x_dom;
        
        % Projection deflation scheme
        A = sparse_deflation(A, x_dom, 'projection', []);
    end
end

function x_dom = TPower_leading_eig(A, x, card)
    % loop parameters
    cont = 1;
    iter = 1;
    iter_max = 2000;
    tolr = 1e-4;
    
    [m, ~] = size(x);
    
    while cont
        % Power method
        xn = A*x;
        xn = xn / norm(xn);
        
        % Compute supp(xn, card)
        [~, idx] = maxk(abs(xn), card);
        
        % Truncation
        xn(setdiff(1:m, idx)) = 0;
        
        % Normalize
        xn = xn / norm(xn);
        
        if sum((xn - x).^2) < tolr
            cont = 0;
        end

        if iter == iter_max
            disp('Maximal iteration achieved.')
            cont = 0;
        end
        
        x = xn;
        iter = iter + 1;
        
    end
    
    x_dom = xn;
end