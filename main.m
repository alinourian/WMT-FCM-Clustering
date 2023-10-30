clear; clc; close all;

load 'data.mat';

scaledData = mat2gray(imageData);
imwrite(scaledData, 'imageData.jpg');

scaledData = mat2gray(imageMask);
imwrite(scaledData, 'imageMask.jpg');

%% ============================= K-Means ==================================
clc; close all;

mask = single(imageMask);
img = single(imageData);
[R, C] = size(mask);

k = 3;
q = 5;
b_const = 1;
b_init = (b_const * ones(R, C)) .* mask;
f = fspecial('gaussian', 9, 2.5);
eps = 1e-7;

[seg, mu, sigma] = KMeans(img, mask, k, eps);

figure;
path = 'res_kmeans';
title_str = 'K-Means Result';
showSegmented(seg, k, title_str, path);

%% =============================== Alg. ===================================
clc;
u = zeros(R, C, k);           % The class memberships
for i=1:k
    u(:, :, i) = (seg == i);
end
b = b_init;        % The bias field
c = mu;            % The class means
q = 5;             % The q-parameter as specified in the slides
w = f;             % The neighbourhood mask
J_init = objectiveFunction(img, b, c, q, u, w);        % The initial value of the loss function
eps = 1e-5;
N_max = 250;
[u, b, c, J] = iterate(img, mask, u, b, c, q, w, J_init, eps, N_max);
bias_removed_image = mask .* img ./ b;
figure;
plot(1:N_max, J);
title(['Objective Function (N=', num2str(N_max), ')']);
grid minor;

figure;
path = 'res_seg';
title_str = 'Segmented Image';
showSegmented(u, 1, title_str, path);
%% ========================================================================
close all;
figure;
imshow(img);
title('Corrupted Image');
figure;
fig = imshow(b);
title('Bias Field');
saveas(fig, 'res_biasRemovedField', "jpg");
figure;
fig = imshow(bias_removed_image);
title('Bias Removed Image');
saveas(fig, 'res_biasRemovedImage', "jpg");
figure;
showSegmented(seg, k, 'K-Means Result', '', false);
figure;
showSegmented(u, 1, 'Segmented Image', '', false);
figure;
fig = imshow(img - mask .* img);
title('Residual Image');
saveas(fig, 'res_residualImage', "jpg");
