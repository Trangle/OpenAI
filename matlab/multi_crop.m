function Is=sample(imgPath)
clc;close all;clear;
imgPath = 'D:\BOT\data\preview\11³¤¾±Â¹\00fb01d68f834e6f91cd352bf8915426.jpg';
Is = crop(imgPath);
end

function Is=crop(imgPath)
shortSides=[224, 256,320,384, 480];
P = 224;

Is = {};
I0 = imread(imgPath);

for k=1:length(shortSides)
    L = shortSides(k);
    
    if size(I0,2)>size(I0,1)
        I = imresize(I0,[L,NaN]);
        full = imresize(I,[NaN,P]);
    else
        I = imresize(I0,[NaN,L]);
        full = imresize(I,[P,NaN]);
    end
    
    H = size(I,1);
    W = size(I,2);
    patch_left_top=getPatch(I,[1,1,P,P]);
    patch_right_top=getPatch(I,[W-P+1,1,P,P]);
    patch_left_bottoom=getPatch(I,[1,H-P+1,P,P]);
    patch_right_bottom=getPatch(I,[W-P+1,H-P+1,P,P]);    
    patch_center=getPatch(I,[floor((W-P)/2+1),floor((H-P)/2+1),P,P]);
    
    patch_full = zeros(P,P,3,'uint8');
    rect = [floor((P-size(full,2))/2+1),floor((P-size(full,1))/2+1),size(full,2),size(full,1)];
    patch_full(rect(2):rect(2)+rect(4)-1,rect(1):rect(1)+rect(3)-1,:)=full;
    
    Is =[Is, patch_left_top];
    Is =[Is, fliplr(patch_left_top)];
    
    Is =[Is, patch_right_top];
    Is =[Is, fliplr(patch_right_top)];
    
    Is =[Is, patch_left_bottoom];
    Is =[Is, fliplr(patch_left_bottoom)];
    
    Is =[Is, patch_right_bottom];
    Is =[Is, fliplr(patch_right_bottom)];
    
    Is =[Is, patch_center];
    Is =[Is, fliplr(patch_center)];
    
    Is =[Is, patch_full];
    Is =[Is, fliplr(patch_full)];
     
%     figure(1),imshow(I)
%     figure(2),imshow(patch_center)
end

end

% I:input image
% rect: crop region,[x,y,w,h]
function patch=getPatch(I,rect)
patch = I(rect(2):rect(2)+rect(4)-1,rect(1):rect(1)+rect(3)-1,:);
end

