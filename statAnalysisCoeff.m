

close all;
clear all;

load('videoFiles.mat');
load('/Users/mcarsala/Documents/MATLAB/1. thesis/Resultados/brisque-train/qoe-qp210.mat', 'model')

oldpath=path;
path(oldpath,'/Users/mcarsala/Documents/MATLAB/ompbox');
oldpath=path;
path(oldpath,'/Users/mcarsala/Documents/MATLAB/ksvdbox');
path(oldpath,'/Users/mcarsala/Documents/MATLAB/1. thesis');


seq="A3";
QP=210; % [85 110 135 160 185 210];
w=1280;
h=720;
sz=32;
fr=1;
maxatoms=1;
n= 1; % number of frames to process
m =500; %random patches per frame
t=1;
umbral1=0;
umbral2=100000000;

%Dictionaries




D=subdict(sensingDictionary(Size=((sz/fr))^2,Type={'dct'}),1:((sz/fr))^2,1:1*(((sz/fr))^2));
%D2=subdict(sensingDictionary(Size=(sz/fr)^2,Type={'dwt'}),1:(sz/fr)^2,1:1*(sz/fr)^2);
%D2=D2./sqrt(sum(D2.^2));
%load('/Users/mcarsala/Documents/MATLAB/My_SRcode_Sparse/Learning_Dictionaries/Dh_16_512.mat')
%D3 = Dh;
%clear Dh;
%load('/Users/mcarsala/Documents/MATLAB/1. thesis/Dictionaries/Universal/Dh_QP210_disto_16_512.mat')
%D3 = [D3 Dh];
%D3=D3./sqrt(sum(D3.^2));

D3= [D];
%load paths

base_path="/Users/mcarsala/phd-results/";
%distorted_qp210_path_noResto= "/Users/mcarsala/phd-results/dictionaryTraining/QP210/noResto/";
%distorted_qp85_path_noResto= "/Users/mcarsala/phd-results/dictionaryTraining/QP85/noResto/";


reference_path = base_path + "aom-sequences/" + seq + "/";
distorted_path = base_path + "refnoResto/decodedYUVs/";



%A2_ref = ["ControlledBurn_1280x720p30_420.yuv","Johnny_1280x720_60.yuv" ....
             % "KristenAndSara_1280x720_60.yuv" "Vidyo3_1280x720p_60fps.yuv" ....
             % "Vidyo4_1280x720p_60fps.yuv" "WestWindEasy_1280x720p30_420.yuv"];

%A2_qp210_noResto_ai = ["ControlledBurn_1280x720p30_420_60_qp210_noResto_ai.yuv","Johnny_1280x720_60_qp210_noResto_ai.yuv" ....
              %"KristenAndSara_1280x720_60_qp210_noResto_ai.yuv" "Vidyo3_1280x720p_60fps_qp210_noResto_ai.yuv" ....
              %"Vidyo4_1280x720p_60fps_qp210_noResto_ai.yuv" "WestWindEasy_1280x720p30_420_qp210_noResto_ai.yuv"];



%A2_qp85_noResto_ai = ["ControlledBurn_1280x720p30_420_60_qp85_noResto_ai.yuv","Johnny_1280x720_60_qp85_noResto_ai.yuv" ....
             % "KristenAndSara_1280x720_60_qp85_noResto_ai.yuv" "Vidyo3_1280x720p_60fps_qp85_noResto_ai.yuv" ....
             % "Vidyo4_1280x720p_60fps_qp85_noResto_ai.yuv" "WestWindEasy_1280x720p30_420_qp85_noResto_ai.yuv"];


ref = referenceContent8bits(referenceContent8bits==seq,2)




p= size(ref,1);

    t=1;
    clear gamma_t;
    clear gamma2;
    clear gamma1;
    
for i=1:p
    %t=1;
    %clear gamma_t;
    %clear gamma2;
    %clear gamma1;
    

    for j=1:n

        reference_yuv = reference_path + ref(i) + ".yuv";
        distorted_yuv = distorted_path + ref(i) + "_aom_av2_AI_Preset_0_QP_" + QP + ".yuv";
        %distorted_yuv ="/Users/mcarsala/phd-results/dictionaryTraining/QP85/noResto/ControlledBurn_1280x720p30_420_60_qp85_noResto_ai.yuv";

        [Y U V]=yuvRead(reference_yuv,w,h,n);
        [Yd Ud Vd]=yuvRead(distorted_yuv,w,h,n);
    
        for k=1:m

            %x_1 = sz+1+round((size(Y,1)-sz^2-sz-1)*rand);
            %y_1 = sz+1+round((size(Yd,1)-sz^2-sz-1)*rand);
            
            x_1 = 1+round((size(Y,1)-sz-1)*rand);
            y_1 = 1+round((size(Yd,1)-sz-1)*rand);
            
            %x_1 = 1+round((32-sz)*rand);
            %y_1 = 1+round((32-sz)*rand);
            % 622, 223
            %y_1 = 219;
            %x_1 = 618;
            
            x=double(Y(x_1:x_1+sz-1,y_1:y_1+sz-1,j));
            y=double(Yd(x_1:x_1+sz-1,y_1:y_1+sz-1,j));
        
            %x=double(Y(290+x_1:290+x_1+sz-1,528+y_1:528+y_1+sz-1,j));
            %y=double(Yd(290+x_1:290+x_1+sz-1,528+y_1:528+y_1+sz-1,j));
            d=x-y;
           
            x_pre = zeros(size(d));
            
            blocksize=(sz/fr);
            blocksize = ones(1,2)*blocksize;
            stepsize = (sz/fr)-1;
            stepsize = ones(1,2)*stepsize;
            epsilon=0.01;
            memusage = 'MEM_LOW';
            nz=0;
            maxval=255;
        
            d=reshape(d,[((sz/fr))^2 1]);
            
            %[blocks_d, dc] = remove_dc(d,'columns');
            blocks_d=d;
            dc = zeros(1,size(blocks_d,2));
            
            gamma_s = omp2(D3,blocks_d,[],epsilon,'maxatoms',maxatoms,'checkdict','on');
            gamma2=double(full(gamma_s));
            %cleanblock = add_dc(D3*abs(gamma2),dc ,'columns');
            cleanblock = add_dc(D3*gamma2,dc ,'columns');
            var_(t)=sum(sqrt(d.^2))./size(d,1);
 
            %%%%%%
            y3=reshape(y,sz^2,1);
            y3=y3./sqrt(sum(y3.^2));
            c3=cleanblock./sqrt(sum(cleanblock.^2));
            [f,x1,x2]=dotvectorDict(y3,D);

            [ax,bx,cx]=find(gamma2);

            dot_y_c=dot(y3,c3);
            var2 =sqrt(sum((y3 - mean(y3)).^2))/sz^2;
        
            if( isempty(ax))
                ax=0;
            end
            if(isempty(cx))    
                cx=0;
            end
            
            a_ = zeros(maxatoms,1);
            b_ = zeros(maxatoms,1);
            c_ = zeros(maxatoms,1);

            [a_,b_,c_]=find(gamma2);
            g1= ones(1024,1);
            g2= ones(1024,1);
            %g1(a(1))=-1;
            %g2(a(2))=-1;

            cleanblock2 = add_dc(D3*(-1)*abs(gamma2),dc ,'columns');
            %cleanblock3 = add_dc(D3*g1.*gamma2,dc ,'columns');
            %cleanblock4 = add_dc(D3*g2.*gamma2,dc ,'columns');
            

            %sim_normal=ssim(y+reshape(cleanblock,sz,sz),:
            % 
            
            %sim_invertido=ssim(y+reshape(cleanblock2,sz,sz),y)

            %[mssim2, ssim_map2] = ssim(y+reshape(cleanblock2,sz,sz),y);
            %[mssim1, ssim_map1] = ssim(y+reshape(cleanblock,sz,sz),y);
            
            %mssim1;
            %mssim2;
            %sim_normal = sqrt(sum(sum(ssim_map1.^2)))/sz^2;
            %sim_invertido = sqrt(sum(sum(ssim_map2.^2)))/sz^2;
            
            gamma_s;

            vary=sum(sqrt(reshape(y -mean(y),sz^2,1).^2))/sz^2;

            %ssim(x,y+reshape(cleanblock3,32,32))
            %ssim(x,y+reshape(cleanblock4,32,32))
            % 119.1570 + 35.6317*rand(1,1)
            %y3=reshape(y,sz^2,1);
            %x3=reshape(x,sz^2,1);
            %y3=y3./sqrt(sum(y3.^2));
            %x3=x3./sqrt(sum(x3.^2));
            %y3=dct(y3);
            %x3=dct(x3);
            %max_x2= max(x2);
            %max_x1= max(x1);
            %absolute = x1;
            %x2= x2;
            %mask = [0.63	0.1269	0.0253	0.0046	0.0018	0.0068	0.0211 ...
            %0.0895	0.0399	0.0325	0.0068	0.0068	3.37E-04	0.0013	0.0032	0.0084];
            %max(mask.*absolute)
            %mask.*absolute

            %res(:,t)=[var2 var(t) reshape(y,sz^2,1)' x2 ax cx csnr(x,y,0,0) csnr(x,y+reshape(cleanblock,sz,sz),0,0)];
            %close all
            %figure(1); plot(2:sz^2,[x2(2:end)']);
            %figure(2); subplot(1,3,1);imshow(x./255,'InitialMagnification','fit')
            %subplot(1,3,2);imshow(y./255,'InitialMagnification','fit')
            %subplot(1,3,3);imshow((y+reshape(cleanblock,sz,sz))./255,'InitialMagnification','fit')
            %res(:,t)= [var2 var(t) x2 ax(1) cx(1) csnr(x,y,0,0) csnr(x,y+reshape(cleanblock,sz,sz),0,0)];
            %%%%%%
            %gamma
            %vary
            %dct_pos_predictor(y,blocks_d,x)
            %gamma_s;
            csnr(x,y,0,0);
            csnr(x,y+reshape(cleanblock,sz,sz),0,0);
            %res(:,t)= [dct_pos_predictor_bk21022023(y,blocks_d,x,QP)];
            res(:,t)= [dct_pos_predictor_f6(y,blocks_d,x,QP,1)];
            %dct_pos_predictor(y,blocks_d,x,QP,1)
            %subplot(1,3,1);imshow(x./255,'InitialMagnification','fit');subplot(1,3,2);imshow(y./255,'InitialMagnification','fit');subplot(1,3,3);imshow((y+reshape(cleanblock,sz,sz))./255,'InitialMagnification','fit')
            psnr_t(t)=csnr(x,y+reshape(cleanblock,sz,sz),0,0);
            multissim_t(t)=multissim(x,y+reshape(cleanblock,sz,sz));
            ssim_t(t)=ssim(x,y+reshape(cleanblock,sz,sz));
            brisque_x_t(t)=brisque(x);
            brisque_y_t(t)=brisque(y);
            brisque_yr_t(t)=brisque(y+reshape(cleanblock,sz,sz));
            %imwrite(cast(x,'uint8'),sprintf('./brisque-train/qp210_%d.bmp',t));
            %gamma_t(:,:,t) =raster(reshape(gamma2,sz,sz));
            gamma_t(:,:,t) =gamma2;
            t=t+1;


   
        end
    end

gamma_t;
%[a2,b2,c2]=find(gamma_t);


[a,b,c]=find(var_>=umbral1 & var_<umbral2);


gamma_t2=gamma_t(:,:,b);
[a2,b2,c2]=find(gamma_t2);

%pdca=fitdist(c2(c2>0),'normal')
%pdca=fitdist(c2(c2>0),'normal')
%mu(i)=pdca.mu;
%sigma(i)=pdca.sigma;

end

[a,b,c]=find(var_>=umbral1 & var_<umbral2);
gamma_t2=gamma_t(:,:,b);

[a2,b2,c2]=find(gamma_t2);
hist(a2,size(D3,2));
%mu_ave=sum(mu)/p
%sigma_ave=sum(sigma)/p

save(strcat("/Users/mcarsala/Documents/MATLAB/1. thesis/Resultados/statistics/res_" +QP +"_D", ...
    num2str(sz^2),'_atoms',num2str(maxatoms),'.mat'),'gamma_t2', 'gamma_t','var_','psnr_t');



