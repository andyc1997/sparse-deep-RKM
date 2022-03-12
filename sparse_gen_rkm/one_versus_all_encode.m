function Ysgn = one_versus_all_encode(Ytr)
    % Ytr: training response for classification, in numeric labels between
    % 0 to L
    % one-versus-all encoding
    Ysgn = 2*dummyvar(Ytr + 1) - 1;
end