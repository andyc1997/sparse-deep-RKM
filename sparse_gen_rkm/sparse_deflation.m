function [Ad, Qt] = sparse_deflation(A, x, scheme, Q)
    % ***********************************************************
    % Deflation methods for sparse PCA (Mackey, 2008)
    % A: covariance matrix for deflation
    % x: sparse eigenvector, should be a column vector
    % scheme: hotelling, projection, Schur complement, orthogonalized
    % hotelling
    % Q: Previous orthonormal basis, only for orthogonalized hotelling
    % scheme
    % Ad: deflated covariance matrix
    % Qt: Updated orthonormal basis, only for orthogonalized hotelling
    % scheme
    % ***********************************************************
    
    assert(size(x, 2) == 1, 'x must be a column vector');
    
    switch scheme
        case 'hotelling'
            xx = x*x';
            Ad = A - xx*(x'*A*x);
            Ad = (Ad + Ad') / 2; % avoid numerical errors that cause asymmetry
            
        case 'projection'
            [m, ~] = size(x);
            P = eye(m) - x*x';
            Ad = P*A*P;
            Ad = (Ad + Ad') / 2;
            
        case 'schur_comp'
            xA = x'*A;
            Ad = A - A*x*xA/(xA*x);
            Ad = (Ad + Ad') / 2;
            
        case 'ortho_hotelling'
            [m, ~] = size(x);
            assert(isempty(Q) == 0, 'Q must be provided');
            
            q = (eye(m) - Q*Q')*x;
            q = q/norm(q);
            qq = q*q';
            
            Ad = A - qq*(q'*A*q);
            Ad = (Ad + Ad') / 2;
            Qt = [Q, q];
            
        otherwise 
            error(['scheme: ', scheme, ' is invalid']);
            
    end
end