function Xgen = gen_new_x(Xtr, K, params, Htr, Hgen, Nr)
    % Htr: learned hidden feature
    % Hgen: sampled hidden feature, when Hgen = Htr, it is same as
    % denoising

    assert(Nr > 0, 'Nr must be positive')
    assert(strcmp(params{3}, 'eta'), 'eta must be provided')
    
    % denoised similarities
    Kgen = K*Htr*Hgen' ./ params{4};
    
    % scaled similarities
    Kgen = normalize(Kgen, 'range', [0, 1]);
    
    for i = 1:size(Kgen, 2)
        th = maxk(Kgen(:, i), Nr);
        Kgen(Kgen(:, i) < th(end), i) = 0;
    end
    
    const = sum(Kgen, 1) + 10E-12;
    
    % generate new x
    Xgen = diag(1./const)*Kgen'*Xtr;

end

% debugged