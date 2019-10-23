EstimatePrior = load('TrainingSamplesDCT_8.mat')
ZigZagFile = readmatrix('Zig_Zag_Pattern.txt');
TrainsampleDCT_FG = EstimatePrior.TrainsampleDCT_FG;
TrainsampleDCT_BG = EstimatePrior.TrainsampleDCT_BG;
% Tasks:
% Loading the Cheetah Image
% Implementing the DCT on it
% Creating a feature vector from the image

% 1. Loading the image

I = imread('cheetah.bmp');
I_origMask = imread('cheetah_mask.bmp');
I = im2double(I);
I_origMask = im2double(I_origMask);
f1 = figure('Name','Original Image');
imshow(I)
% The size of the image is 255x270

% 2. Implementing the DCT on it.
% padding the image

padded = zeros(262,277);
padded(4:258,4:273) = I;
imshow(I)

% make a slice, take its DCT, create a vector, extract the feature
features = zeros(255*270,1);
jump=1;
for i = 1:255
    for j = 1:270
        slice = padded((1+(jump*(i-1))):(1+(jump*(i-1)))+7,(1+(jump*(j-1))):7+(1+(jump*(j-1))));
        all_cells{i,j} = DCT(slice);
        features(((j-1)*255)+i) = ZigZagify(all_cells{i,j},ZigZagFile);
    end
end

% Right now the training set has vectors, we must convert them into scalars. For this we
% will write a function 'SecondGreatestDCT'.
FG_scalars = zeros(250,1);
BG_scalars = zeros(1053,1);
for i = 1:250
    FG_scalars(i) = SecondGreatestDCT(TrainsampleDCT_FG(i,:));
end
for i = 1:1053
    BG_scalars(i) = SecondGreatestDCT(TrainsampleDCT_BG(i,:));
end


% Now that we have determined X for each pixel in the image, we
% determine the prior probabilities, P(Y), from the the training data,
% namely the matrix TrainingSamplesDCT_8.mat

prior_probability_FG = 250/(250+1053)
prior_probability_BG = 1053/(250+1053)



% Now we plot the index histograms of P(X=x|Y=cheetah) using the TrainsampleDCT_FG
f2 = figure('Name', 'Histogram of probabilities: PX|Y(x|cheetah)');
h1_cheetah = histogram(FG_scalars,1:65,'Normalization', 'probability');
% Now we plot the index histograms of P(X=x|Y=background) using the TrainsampleDCT_FG
f3 = figure('Name', 'Histogram of probabilities: PX|Y(x|background)');
h2_bg = histogram(BG_scalars,1:65, 'Normalization', 'probability');

% We try computing the state variable for each pixel of the cheetah.bmp
% image
all_states = zeros(68850,1);
for i = 1:68850
    current_feature = features(i);
    p_XgivChe = h1_cheetah.Values(current_feature+1);
    p_che = prior_probability_FG;
    p_XgivBG = h2_bg.Values(current_feature+1);
    p_bg = prior_probability_BG;
    if((p_XgivChe*p_che)>(p_XgivBG*p_bg))
        all_states(i) = 1;
    else
        all_states(i) = 0;
    end
end

B = reshape(all_states, 255,270);
f4 = figure('Name', 'Mask obtained after applying algorithm')
imagesc(B)
colormap(gray(255))
xorred = xor(B,I_origMask);
S = sum(xorred,'all');
Prob_error = S/(270*255)



function [OutMat] = DCT(RawMat)
%DCT Calculates the 2D Discrete Cosine Transform of the input image
OutMat = dct2(RawMat);
end



function [second_highest] = ZigZagify(eight_square_mat, ZigZagMat)
%ZigZagify converts an 8 by 8 array into a vector of size 64x1 and returns
%the position of its second largest element
zigzag_vector = zeros(64,1);
for i = 1:8
    for j = 1:8
        zigzag_vector(ZigZagMat(i,j)+1) = eight_square_mat(i,j);
    end
end
zigzag_vector_absolute = abs(zigzag_vector(2:64)); % takes absolute value of 64x1 vector
[ordered_zigzagV,sorted_Indices] = sort(zigzag_vector_absolute, 'descend'); % sorts the absolute vector and returns both the sorted vector as well as a vector of indices
second_highest = sorted_Indices(1); % try playing around with this and see how 'features' changes

end

function [scalar] = SecondGreatestDCT(Vector)
%SecondGreatestDCT Turns Vector with 64 dimensions to scalar and returns it
V_abs = abs(Vector);
[ordered_zigzagV,sorted_Indices] = sort(V_abs, 'descend'); % sorts the absolute vector and returns both the sorted vector as well as a vector of indices
scalar = sorted_Indices(2); % try playing around with this and see how 'features' changes
end

