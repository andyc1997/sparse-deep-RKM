function [Xtr, Xts, Ytr, Yts] = read_mnist(seed, N_subset, N_valid, path_folder)
    % *******************************************************
    % seed: for sampling train and test data
    % path_folder: path to folder storing MNIST dataset
    % N_subset: number of training data points
    % N_valid: number of validation data points (optional, set 0 for
    % unsupervised setting)
    % 
    % Function for reading MNIST data.
    % *******************************************************
    
    % read data
    images = loadMNISTImages([path_folder, '\train-images.idx3-ubyte']);
    labels = loadMNISTLabels([path_folder, '\train-labels.idx1-ubyte']);
    
    % set seed
    rng(seed, 'twister');
    idx_tr = datasample(1:size(images, 2), N_subset, 'Replace', false);
    idx_ts = datasample(setdiff(1:size(images, 2), idx_tr), N_valid, 'Replace', false);
    
    Xtr = images(:, idx_tr)'; % train set
    Xts = images(:, idx_ts)'; % test set
    Ytr = labels(idx_tr); % train set
    Yts = labels(idx_ts); % train set
    
end

% src: https://github.com/davidstutz/matlab-mnist-two-layer-perceptron/blob/master/loadMNISTImages.m
function images = loadMNISTImages(filename)
    fp = fopen(filename, 'rb');
    assert(fp ~= -1, ['Could not open ', filename, '']);

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2051, ['Bad magic number in ', filename, '']);

    numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
    numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
    numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

    images = fread(fp, inf, 'unsigned char');
    images = reshape(images, numCols, numRows, numImages);
    images = permute(images,[2 1 3]);

    fclose(fp);

    % Reshape to #pixels x #examples
    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));

    % Convert to double, normalize
    images = double(images) / 255.0;
end

function labels = loadMNISTLabels(filename)
    %loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
    %the labels for the MNIST images

    fp = fopen(filename, 'rb');
    assert(fp ~= -1, ['Could not open ', filename, '']);

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2049, ['Bad magic number in ', filename, '']);

    numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
    labels = fread(fp, inf, 'unsigned char');
    assert(size(labels,1) == numLabels, 'Mismatch in label count');

    fclose(fp);

end

