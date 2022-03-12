function [Htr, b] = cl_LSSVM(Xtr, Ytr, params)
    % ***********************************************************
    % param = {'rbf_func', sig, 'nclass', integer, 'eta', double, 
    % 'lambda', double}
    % ***********************************************************
    sig = params{2};
    nclass = params{4};
    assert(sig > 0, 'sig must be greater than 0');
    assert(nclass > 0, 'nclass must be greater than 0');
    
    [N, ~] = size(Xtr);
    
    % build kernel
    Kparams = {'rbf_func', sig};
    K = build_kernel(Xtr, Kparams);
    
    % kernel centering
    P = eye(N) - ones(N)/N;
    K = P*(K/params{6})*P;
    
    % coef matrix
    M = [K + params{8}*eye(N), ones(N, 1);
        ones(1, N), 0];
    d = [Ytr; 
        zeros(1, nclass)];
    
    % Solve linear system
    Soln = M\d;
    
    % Return solution
    Htr = Soln(1:end-1, :)';
    b = Soln(end, :)';
end