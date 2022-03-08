function Hgen = gen_latent(Htr, l, Ngen)
    % Htr: latent variable for training data
    % l: number of cluster for GMM
    assert(l > 0, 'l must be positive');
    assert(Ngen > 0, 'Ngen must be positive');
    
    % Generate new latent variables
    GMModel = fitgmdist(Htr, l);
    h_dist = gmdistribution(GMModel.mu, GMModel.Sigma);
    Hgen = random(h_dist, Ngen);
    
end

% debugged