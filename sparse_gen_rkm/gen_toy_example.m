cd 'C:\Users\user\Documents\Statistics\SPCA-DRKM-2021JUL01\work\new-function\sparse_deep_rkm\sparse_gen_rkm'

% toy example in Generative Restricted Kernel Machines: 
% A Framework for Multi-view Generation and Disentangled Feature Learning
% (Pandey et al., 2021)

% Simulate 3 Gaussian distributions
mu1 = [1, 2]; var1 = eye(2);
mu2 = [-3, 0]; var2 = eye(2);
mu3 = [2, -4]; var3 = eye(2);

Nsize = 500;

rng(80116)
grp1 =  mvnrnd(mu1, var1, Nsize);
grp2 =  mvnrnd(mu2, var2, Nsize);
grp3 =  mvnrnd(mu3, var3, Nsize);

% plot
plot(grp1(:, 1), grp1(:, 2), 'b.', ...
    grp2(:, 1), grp2(:, 2), 'k.', ...
    grp3(:, 1), grp3(:, 2), 'r.')
xlabel('First coordinate $x$', 'interpreter', 'latex')
ylabel('Second coordinate $y$', 'interpreter', 'latex')
title('Toy example with 3 Gaussian distributions')

% export data
toy = [grp1, ones(Nsize, 1); ...
    grp2, 2.*ones(Nsize, 1); ...
    grp3, 3.*ones(Nsize, 1)];
save('toy.mat','toy');


