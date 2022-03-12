function Ypred = pred_LSSVM(x, Xtr, Htr, b, params)
    % **************************************************
    % x: point for prediction
    % **************************************************
    k = exp(-sum((Xtr - x).^2, 2)/(2*params{2}^2));
    Ypred = sign(Htr*k/params{6} + b)';
end