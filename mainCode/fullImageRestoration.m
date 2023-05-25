function [restored_yuv gain_psnr psnr_res_ave psnr_dis_ave total_bytes_sparse blocknum] = fullImageRestoration16(frame,quantization)

clear all;
close all;

qt=[110 135 160 185  210 85];
for qq=1:1
for q=5:5

    oldpath=path;
path(oldpath,'/Users/mcarsala/Documents/MATLAB/ompbox');
oldpath=path;
path(oldpath,'/Users/mcarsala/Documents/MATLAB/ksvdbox');

oldpath=path;
path(oldpath,'/Users/mcarsala/Documents/MATLAB/YUV/YUV');

oldpath=path;
path(oldpath,'/Users/mcarsala/Documents/MATLAB/1.thesis');

addpath('../');
addpath('/Users/mcarsala/Documents/MATLAB/1. thesis/histfitlaplace')
load('videoFiles.mat');
load("sparseParameters_v1.mat");
load('bitCodingP.mat')
% parameters
sparse_resto=1;
sparse_resto_coeff=1;
sparse_resto_pos=1;
frame =qq;
n=5; % number of frames
seq ="A2";
w=1920;
h=1080;
QP=qt(q); %[110 135 160 185  210 85]
ru = 1;
fps=30;
sz=32;  % Block size
fr=1;  %  Up/down sampling 
block=sz; % block size
step =sz-1; % step
maxatoms=1;  % Max atom per vector

if(QP==185)
    %factor=0.6;
    factor=0.8;
elseif (QP==210)
    %factor=0.35;
    factor=0.35
else
    factor=1;
end
bits_dis=0;

epsilon=1; % error



base_path="/Users/mcarsala/phd-results/";

reference_path = base_path + "aom-sequences/" + seq + "/";
reference_restoav2_path = base_path + "refResto/decodedYUVs/";
distorted_path = base_path + "refnoResto/decodedYUVs/";


ref = referenceContent8bits(referenceContent8bits==seq,2);


restored_yuv= "test_rest4.yuv";

reference_yuv = reference_path + ref(frame) + ".yuv";
restored_av2= reference_restoav2_path + ref(frame) + "_aom_av2_AI_Preset_0_QP_" + QP + ".yuv";
distorted_yuv = distorted_path + ref(frame) + "_aom_av2_AI_Preset_0_QP_" + QP + ".yuv";


delete(restored_yuv);
delete("temporal_ref.yuv");
delete("restored_av2.yuv");
delete("temporal_dis.yuv");


for i=1:n

[Y2 U2 V2 ]=yuvRead(reference_yuv,w,h,i);

[Y3 U3 V3 ]=yuvRead(distorted_yuv,w,h,i);

[Y4 U4 V4 ]=yuvRead(restored_av2,w,h,i);


y_ori=double(Y3(:,:,i));
x_ori=double(Y2(:,:,i));



h=size(x_ori,1);
w=size(y_ori,2);


[str_l str_r str_t str_b] = calcstripe (w,h,block,step);



yt = [repmat(y_ori(:,1,1),1,str_l) y_ori(:,:,1) repmat(y_ori(:,end,1),1,str_r)];
yt2 =[repmat(yt(1,:),str_t,1); yt(:,:); repmat(yt(1,:),str_b,1)];     

y =double(yt2);



xt = [repmat(x_ori(:,1,1),1,str_l) x_ori(:,:,1) repmat(x_ori(:,end,1),1,str_r)];
xt2 =[repmat(xt(1,:),str_t,1); xt(:,:); repmat(xt(1,:),str_b,1)];     


x=double(xt2);


d=x-y;


D=subdict(sensingDictionary(Size=(sz/fr)^2,Type={'dct'}),1:(sz/fr)^2,1:1*(sz/fr)^2);
D3 = [D];



x_pre = zeros(size(x));
blocksize = ones(1,2)*block;
stepsize = ones(1,2)*step;

memusage = 'MEM_LOW';
nz=0;
lambda=0;
maxval=255;
realblocks=0;
bits_sum =0;
blocknum = prod(floor((size(x)-blocksize)./stepsize) + 1);
blocknum_i=prod(floor((size(x)-blocksize)./stepsize) + 1)
x_cnt = countcover(size(d),blocksize,stepsize);


t=0;
%tic
for j = 1:stepsize(2):size(x_pre,2)-blocksize(2)+1

  clear gamma2;
  % the current batch of blocks
  blocks_d2 = im2colstep(d(:,j:j+blocksize(2)-1),blocksize,stepsize);
  x_2 = im2colstep(x(:,j:j+blocksize(2)-1),blocksize,stepsize);
  y_2 = im2colstep(y(:,j:j+blocksize(2)-1),blocksize,stepsize);

  cleanblocks=zeros(size(blocks_d2));
  cleanblocks_n=zeros(size(blocks_d2));


  blocks_d3=double(imresize(blocks_d2,[size(blocks_d2,1)/(fr^2) size(blocks_d2,2)],'bicubic'));
  % remove DC

  %[blocks_d, dc] = remove_dc(blocks_d3,'columns');
  dc = zeros(1,size(blocks_d2,2));
  blocks_d=blocks_d3;
  %%%%%
  gamma = omp2(D3,blocks_d,[],epsilon,'maxatoms',maxatoms,'checkdict','on');
  
  gamma2=double(full(gamma));
  
  gamma3=gamma2;




%
%%%%%%%%  coeff sparse aleatorios


clear a;
clear b;

f_dis_f = zeros(size(gamma3,2),1);
if (sparse_resto==1)

    if (sparse_resto_coeff == 1)


    %%%%%%%%
    end


  for wx=1:size(blocks_d,2)

        cleanblocks_n(:,wx) = dct_pos_predictor_f1(reshape(y_2(:,wx),sz,sz),blocks_d(:,wx),reshape(x_2(:,wx),sz,sz),QP,1);
        % REcuerda cambiar el criterio de skip de acuerdo al criterio%

  end
  
end

%%%%

  gamma_t(:,:,t+1) =gamma3;
 


  cleanblocks=cleanblocks_n;


  
  nz = nz + nnz(gamma3);
  

   psnr_v = zeros(size(blocks_d2,2),1);
   ssim_v = zeros(size(blocks_d2,2),1);

  
  for z=1:size(blocks_d2,2)

   psnr_v(z)= csnr(y_2(:,z)+cleanblocks(:,z),x_2(:,z),0,0)-csnr(y_2(:,z),x_2(:,z),0,0);
   ssim_v(z)= ssim(y_2(:,z)+cleanblocks(:,z),x_2(:,z))-ssim(y_2(:,z),x_2(:,z));
   var(z)=sum(abs(blocks_d3(:,z)))./size(blocks_d3,1);

   
    if (psnr_v(z)<=0)

       sprintf('Delta PSNR: %.2fdB',  psnr_v(z));

       nz = nz - nnz(gamma3(:,z));
       blocknum = blocknum-1;
       bits_dis(z)=1;
       cleanblocks(:,z)=zeros(size(blocks_d2,1),1);
            for ii=1+(z-1)*stepsize(2):1+(z-1)*stepsize(2)+ blocksize(2)-1
               for iii=j:j+blocksize(2)-1
       
                    if x_cnt(ii,iii) >1
                        x_cnt(ii,iii)= x_cnt(ii,iii)-1;
                    end
                end
           end
    end
  end
  
  bits_sum = bits_sum+sum(bits_dis);
  psnr_(:,t+1) =psnr_v;

  
  cleanim = col2imstep(cleanblocks,[size(x_pre,1) blocksize(2)],blocksize,stepsize);

  x_pre(:,j:j+blocksize(2)-1) = x_pre(:,j:j+blocksize(2)-1) + cleanim;

t=t+1;
end
%enc(i) = toc

nz = nz/blocknum; % 
%cnt = countcover(size(d),blocksize,stepsize);

%x_pre = (x_pre+lambda*d)./(cnt + lambda); % For overlapping

x_pre = (x_pre)./(x_cnt); % For overlapping
 


x_pre= x_pre+y;

x_pre_f(1:size(x_ori,1),1:size(x_ori,2)) = x_pre(str_t+1:end-str_b,str_l+1:end-str_r);

yuv_export(cast(x_pre_f,'uint8'),U3(:,:,i),V3(:,:,i),restored_yuv,1);
yuv_export(cast(y_ori,'uint8'),U3(:,:,i),V3(:,:,i),"temporal_dis.yuv",1);
yuv_export(cast(x_ori,'uint8'),U2(:,:,i),V2(:,:,i),"temporal_ref.yuv",1);
yuv_export(Y4(:,:,i),U3(:,:,i),V3(:,:,i),"restored_av2.yuv",1);

subplot(3,1,1); imshow(x_ori/maxval);
title('Reference image','fontsize', 18,'interpreter','latex');

subplot(3,1,2); imshow(y_ori/maxval); 
title(sprintf('Decoded (distorted) image, PSNR = %.2fdB', 20*log10(maxval * sqrt(numel(x_ori)) / norm(x_ori(:)-y_ori(:))) ),'fontsize', 18,'interpreter','latex');


subplot(3,1,3); imshow(x_pre_f/maxval);
title(sprintf('Restored image, PSNR: %.2fdB', 20*log10(maxval * sqrt(numel(x_ori)) / norm(x_ori(:)-x_pre_f(:))) ),'fontsize', 18,'interpreter','latex');

psnr_dis(i) = csnr(x_ori,y_ori,0,0);
ssim_dis(i) = ssim(y_ori,x_ori);

psnr_res(i) = csnr(x_ori,x_pre_f,0,0);
ssim_res(i) = ssim(x_pre_f,x_ori);
blocknum_t(i)=blocknum;
bits_sum_t(i) = bits_sum;

end



factor = 256/size(D,2);
bits_coeff_sig=1; %total_bytes_for_signaling_coeff
bits_coeff_pos_sig=3;
total_bytes_sparse= blocknum*(maxatoms* (bits_coeff_sig+bits_coeff_pos_sig))/8;
%total_bytes_sparse = (fr^2)*maxatoms + (fr^2)*(1/))

reference_wiener = 480; %  2(int16)x2(v/h filters)x3(coffecientes)x15 (RU)

%psnr_referencia =  0.1186

%gain_psnr =[psnr_dis' psnr_res'];


psnr_dis_ave= sum(psnr_dis)/size(psnr_dis,2)

psnr_res_ave= sum(psnr_res)/size(psnr_res,2);

ssim_dis_ave= sum(ssim_dis)/size(ssim_dis,2)

ssim_res_ave= sum(ssim_res)/size(ssim_res,2);

gain_psnr = psnr_res_ave - psnr_dis_ave;

blocknum_ave= round(sum(blocknum_t)/size(blocknum_t,2));
bits_sum_ave= sum(bits_sum_t)/size(bits_sum_t,2);
%encT= sum(enc)/size(enc,2)



%system("/opt/homebrew/bin/vmaf -w " + w + " -h " + h + " -b 8  -p 420  -m path=" + '"/Users/mcarsala/vmaf/model/vmaf_float_v0.6.1.json" -r ' + "temporal_ref.yuv" + " -d " + "temporal_dis.yuv")
%system("/opt/homebrew/bin/vmaf -w " + w + " -h " + h + " -b 8  -p 420  -r " + "temporal_ref.yuv" + " -d " + "temporal_dis.yuv")
%system("/opt/homebrew/bin/vmaf -w " + w + " -h " + h + " -b 8  -p 420  -r " + "temporal_ref.yuv" + " -d " + "restored_av2.yuv")
[status,cmdout] = system("/opt/homebrew/bin/vmaf -w " + w + " -h " + h + " -b 8  -p 420  -r " + "temporal_ref.yuv" + " -d " + restored_yuv);

pattern='vmaf_v0.6.1:\s([\d.]+)';


vmaf=regexp(cmdout, pattern, 'match');

sprintf('%s, %d, %d, %d, %f , %s, %f, %d',restored_av2,frame,QP, blocknum_ave, psnr_res_ave,vmaf{1},ssim_res_ave,(blocknum_ave+blocknum_i)*30/1024)

sprintf('%f,%d',psnr_res_ave,round(bits_sum_ave*fps/1024))

end
end
return

