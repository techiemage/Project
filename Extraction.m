clc;
close all;
clear all;
load data.mat
watermarked_Image=imread('Watermarked.bmp');
figure,imshow(uint8(watermarked_Image));
title('Watermarked Image')
[rows,columns,c]=size(watermarked_Image);
block_width = 4;
block_height = 4;
[width,height] = size(watermarked_Image);
OrgImg= zeros(width, height);
grid_width = width / block_width;
grid_height = height / block_height;
k=1;
fstart=1;
fend=4;
kdata=1;
Retrived_Image = zeros(width, height);
extractedBitStream=MsgBitStream;
for gx = 1:grid_width
    for gy = 1:grid_height
        cx = (gx - 1) * block_width + 1;
        cy = (gy - 1) * block_height + 1;
        posx = cx:cx + block_width - 1;
        posy = cy:cy + block_height - 1;
        block = watermarked_Image(posx, posy);
        [LL, HL, LH, HH] = Slantlet2D(block);
        [U, S, V] = svd(LL);
        if(fend<=length(MsgBitStream))
        for ii=1:block_width-2
            for jj=1:block_height-2
                singular_value = S(ii, jj);
                if mod(round(singular_value), 2) == 0
                    extractedBitStream(k) = 1; 
                    singular_value = singular_value-1;
                end
                
            k=k+1;
            kdata=kdata+1;
            end
        end
        S=singular_value;
        fstart=fstart+4;
        fend=fend+4;
        LL0 = U * S * V';
        Retrive_block = iSlantlet2D(LL0, HL, LH, HH);
        Retrived_Image(posx, posy) = Retrive_block(:,:);
        else
        Retrived_Image(posx, posy) = block(:,:);
        end
    end
end
% Display the retrieved bits
MsgBitStreamReverse = extractedBitStream.*MsgBitStream;
finalImage =reshape(MsgBitStreamReverse,[32,32]);
figure,imshow(finalImage);
title('Extracted OwnerShip Share')
figure,imshow(uint8(Retrived_Image));
title('Restored Image')
RW = reshape(finalImage,1,Wh*Ww);
R = 220; % Radius
N = 44;  % Number of rings
M = 16;  % Number of fan rings
BIF = feature_extraction_NNGR(uint8(Retrived_Image),R,N,M);
h0 = input('Enter key between 0.2 and 1\n');
u = 3.9;
v = 10;
L=M;
num_iterations = (2 * (N - 1) * L)+16;
h = zeros(num_iterations, 1);
h(1) = h0;
% Generate chaotic sequence
for n = 1:num_iterations-1
    h(n+1) = Echaos(u, h(n), v);
end
% Binarize the chaotic sequence
T = mean(h);
BH = h > T;
% XOR operation
F1 = bitxor(BIF, BH);
L = Wh*Ww;
O = zeros(L, 1);
for i = 1:L
    O(i) = bitxor(F1(i), RW(i));
end
finalImage=reshape(O,[32,32]);
figure,imshow(finalImage);
title('Watermark Information Recovery'); 
disp('================= Watermark Authentication =========================')
MSE=0.0; 
for K=1:1
for i=1:32
     for j=1:32
         MSE=MSE+abs(double(Imgsecret(i,j,K))-double(finalImage(i,j,K)))^2;
     end
end
end
MSE=MSE/(rows*columns);
PSNR=10.0*log10((255.0*255.0)/MSE);
disp('The Mean Square Error is')
disp(MSE);
disp('The PSNR is')
disp(PSNR);
if(MSE>0)
    disp('=== Authentication Failed ====')
else
    disp('=== Authentication Success ====')
end

