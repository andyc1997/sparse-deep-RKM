function score = fid_score(Xgen, Xtr)
    % *************************************************************
    % Xgen: generated data
    % Xtr: training data
    %
    % score = Frechet Inception Distance, assess the quality of images
    % by comparing the distributions
    % *************************************************************
    
    % mean for two data sources
    mean_gen = mean(Xgen, 1);
    mean_tr = mean(Xtr, 1);
    
    % covariance for two data sources
    cov_gen = cov(Xgen);
    cov_tr = cov(Xtr);
    
    % calculate FID score
    score = norm(mean_gen - mean_tr)^2 + trace(cov_gen + cov_tr - 2*real(sqrtm(cov_gen*cov_tr)));
    
end