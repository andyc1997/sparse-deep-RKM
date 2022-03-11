function V = mm_sparse_eigen(A, b, s, rhos)
    % ***********************************************************
    % MM l0 algortihm
    % Sparse Generalized Eigenvalue Problem Via Smooth Optimization
    % 
    % A: covariance matrix
    % b: vector of all positive elements
    % s: number of components to be extracted
    % rho: penalty term for l0 norm
    % 
    % max_{x} x'*A*x - rho*card(x)
    % s.t. x'*diag(b)*x = 1
    % ***********************************************************
    assert(all(rhos > 0), 'rho must be positive');
    
    % if b is not provided, assume an identity matrix for diag(b)
    if isempty(b) == 1
        [nb, ~] = size(A);
        b = ones(nb, 1);
    end
    
    A = diag(1./sqrt(b))*A*diag(1./sqrt(b));
    [m, ~] = size(A);
    V = zeros(m, s);
    
    for comp = 1:s
        % loop parameters
        iter = 1;
        iter_max = 2000;
        tolr = 1e-4;
        cont = 1;
        rho = rhos(comp);
        
        % get new data matrix
        [nV, nD] = eig(A, 'balance');
        C = nV*sqrt(abs(nD))*nV';
        
        % initialized x0, same as GPower method, proposed by Journee et al.
        % (2010)
        [rho_max, i_max] = max(sqrt(sum(C.^2, 1)));
        x = C(:, i_max) / norm(C(:, i_max));
        
        % unnormalize rho
        rho = rho * rho_max^2;
        % disp(rho_max)
        
        while cont
            % holder for the next iterate
            xn = zeros(m, 1);

            % Sort a in descending abs values
            a = 2*A*x;
            [~, idx] = sort(abs(a), 'descend');
            mapidx = [(1:m)', idx]; % for reordering
            mapidx = sortrows(mapidx, 2);
            a_sorted = a(idx);

            % rho too large, prune to 0
            if abs(a_sorted(1)) <= rho
                xn(1) = sign(a_sorted(1));

            else 
                % binary search for proposition 6
                l = 1;
                r = m;
                not_found = 1;

                while not_found
                    mid = floor((l + r)/2);
                    if aux_func(mid, a_sorted, rho) < 0
                        l = mid;
                    else
                        r = mid;
                    end

                    if l + 1 == r 
                        not_found = 0;
                        % disp(['l: ', num2str(l), ' r: ', num2str(r), ' mid: ', num2str(mid)])
                        % disp(['l+1: ', num2str(aux_func(l+1, a_sorted, rho)), ' l: ', num2str(aux_func(l, a_sorted, rho))])
                    end

                end

                assert(aux_func(l+1, a_sorted, rho) >= 0 && aux_func(l, a_sorted, rho) < 0, 'problem in binary search')
                assert(l > 0, 'l cannot be zero');

                % update new sparse eigenvector
                xn(1:l) = a_sorted(1:l);
                xn = xn / norm(a_sorted(1:l));
            end

            % reorder new sparse eigenvector
            xn = xn(mapidx(:, 1));

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

        % reconstruct V
        V(:, comp) = diag(1./sqrt(b))*x;
        
        % Schur complement deflation scheme
        P = eye(m) - x*x';
        A = P*A*P;
        A = (A' + A)/2; % avoid numerical errors that cause asymmetry
    end
    
end

function FVAL = aux_func(p, a_sorted, rho)
    % proposition 6
    FVAL = norm(a_sorted(1:p - 1)) + rho - norm(a_sorted(1:p));
end