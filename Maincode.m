
clc
clear all
close all
%------------ Image Reading ------------------------------------------
[FILENAME,PATHNAME]=uigetfile('*','Select the Image');
FilePath=strcat(PATHNAME,FILENAME);
disp('The Image File Location is');
disp(FilePath);
OriginalDataArray=dicomread(FilePath);
OriginalDataArray=imresize(OriginalDataArray,[512,512]);
DataArray = uint8(255 * mat2gray(OriginalDataArray));
figure,imshow(DataArray);
title('Original  image');
[rows,columns,c]=size(DataArray);
%==========Feature Extraction based on NNGR
R = 220; % Radius
N = 44;  % Number of rings
M = 16;  % Number of fan rings
BIF = feature_extraction_NNGR(DataArray,R,N,M);
% Generate scrambled binary features using LLS
% Key Generation
h0  = input('Enter key between 0.2 and 1\n');
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
F = bitxor(BIF, BH);
% Step 3: Generate Ownership Share
[FILENAME,PATHNAME]=uigetfile('*.bmp','Select the Image');
FilePath=strcat(PATHNAME,FILENAME);
disp('The Image File Location is');
disp(FilePath);
Imgsecret=imresize(imread(FilePath),[32 32]);
figure,imshow(Imgsecret);
title('Secret Image');
[Wh,Ww]=size(Imgsecret);
L = Wh*Ww;
W = reshape(Imgsecret,1,Wh*Ww);
O = zeros(L, 1);
W=double(W);
for i = 1:L
    O(i) = bitxor(F(i), W(i));
end
W = reshape(O,[Wh Ww]);
Adj=1;
fO = zeros(1,L);
fO(1,:)=O;
logoImageArray=fO;
[Wh,Ww]=size(logoImageArray);
TotalBitstobeEmbed=Wh*Ww;
block_width = 4;
block_height = 4;
[width,height] = size(DataArray);
watermarked_Image = zeros(width, height);
grid_width = width / block_width;
grid_height = height / block_height;
fstart=1;
fend=4;
flag1=0;
kdata=1;
MsgBitStream=logoImageArray;
finalImage =reshape(MsgBitStream,[32,32]);
figure,imshow(finalImage);
title('OwnerShip Share');

flgemb=zeros(1,length(MsgBitStream));
for gx = 1:grid_width
    for gy = 1:grid_height
        cx = (gx-1) * block_width + 1;
        cy = (gy-1) * block_width + 1;
        posx = cx:cx+block_width-1;
        posy = cy:cy+block_height-1;
        block = DataArray(posx, posy);
        if(fend<=length(MsgBitStream))
        E=MsgBitStream(fstart:fend);
        flag1=flag1+1;
        k=1;
        % Apply SLT and SVD on block
        [LL, HL, LH, HH] = Slantlet2D(block);
        [U, S, V] = svd(LL);
        for ii=1:block_width-2
            for jj=1:block_height-2
                 if(E(1,k) == 1)
                     S(ii,jj)=(round(S(ii,jj))+1) - mod(round(S(ii,jj)),2);
                 else
                     S(ii,jj)=(round(S(ii,jj))-1) - mod((round(S(ii,jj))+1),2);
                 end
            k=k+1;
            kdata=kdata+1;
            end
        end
        fstart=fstart+4;
        fend=fend+4;
        LL0 = U * S * V';
        watermarked_block = iSlantlet2D(LL0, HL, LH, HH);
        watermarked_Image(posx, posy) = watermarked_block(:,:);
        else
        watermarked_Image(posx, posy) = block(:,:);
        end
        
    end
    
end
figure,imshow(uint8(watermarked_Image));
title('Watermarked Image')
save data.mat TotalBitstobeEmbed MsgBitStream Wh Ww Imgsecret
imwrite(uint8(watermarked_Image),'Watermarked.bmp');
MSE=0.0; 
for K=1:1
for i=1:rows
     for j=1:columns
         MSE=MSE+abs(double(DataArray(i,j,K))-double(watermarked_Image(i,j,K)))^2;
     end
end
end
MSE=MSE/(rows*columns);
PSNR=10.0*log10((255.0*255.0)/MSE);
disp('The Mean Square Error is')
disp(MSE);
disp('The PSNR is')
disp(PSNR);

