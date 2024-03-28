for o=21:1:28
    add5='C:\atrca\s';
    add6=num2str(o);
    add7='.mat';
    file=strcat(add5,add6,add7);    
fprintf('Results of the atrca.\n');
load(file);
a=a(:,:,165:1164,:);
data=a;
% List of stimulus frequencies
list_freqs = [8:1:15 8.2:1:15.2 8.4:1:15.4 8.6:1:15.6 8.8:1:15.8];
                                        
% The number of stimuli
num_targs = length(list_freqs);    

% Labels of data
labels = [1:1:num_targs];   
fb_coefs = [1:5].^(-1.25)+0.25;
aa=zeros(1,2800);
b=zeros(1,2800);
for loocv_i=1:1:6
   testdata1 = squeeze(data(:, :, :, loocv_i));
   traindata1 = data;
   traindata1(:, :, :, loocv_i) = [];

for targ_i=1:1:40
    group=zeros(1,1);
    eeg=squeeze(testdata1(targ_i,:,:));
    
    temp=1;
    e=1;
    x=0.2;
    %分配初始两次特征
    for t=0.2:0.05:0.25
    sampl=round(t*250);
    test_tmp=eeg(:,1:sampl);
    traindata=traindata1(:,:,1:sampl,:);
    model = train_trca(traindata, 250, 5);
    for fb_i = 1:1:model.num_fbs
        testdata = filterbank(test_tmp, model.fs, fb_i);
        for class_i = 1:1:model.num_targs
            traindata =  squeeze(model.trains(class_i, fb_i, :, :));
                w = squeeze(model.W(fb_i, class_i, :));
           %原始
            r_tmp = corrcoef(testdata'*w, traindata'*w);
            r(fb_i,class_i) = r_tmp(1,2);
          % [~,~,r_tmp] = canoncorr(testdata'*w, traindata'*w);
          %   r(fb_i, class_i) = r_tmp(1,1);
        end % class_i
    end % fb_i
     rho = fb_coefs*r;
     rho=softmax(rho')';
     for k=1:1:40
         if k==1
             b(temp)=rho(k);
         else
             b((k-1)*70+temp)=rho(k);
         end
     end
     temp=temp+1;
    end
    
    %分配初始两次时间
    
for i=1:2
    for j=1:1:40
        aa((j-1)*70+e)=x;    
    end
    x=x+0.05;
    e=e+1;
end


ttt=0.3;
while ttt<3

group=[ones(1,e),2*ones(1,e),3*ones(1,e),4*ones(1,e),5*ones(1,e),6*ones(1,e),7*ones(1,e),8*ones(1,e),9*ones(1,e),10*ones(1,e),11*ones(1,e),12*ones(1,e),13*ones(1,e),14*ones(1,e),15*ones(1,e),16*ones(1,e),17*ones(1,e),18*ones(1,e),19*ones(1,e),20*ones(1,e),21*ones(1,e),22*ones(1,e),23*ones(1,e),24*ones(1,e),25*ones(1,e),26*ones(1,e),27*ones(1,e),28*ones(1,e),29*ones(1,e),30*ones(1,e),31*ones(1,e),32*ones(1,e),33*ones(1,e),34*ones(1,e),35*ones(1,e),36*ones(1,e),37*ones(1,e),38*ones(1,e),39*ones(1,e),40*ones(1,e)];

    sampl=round(ttt*250);
    test_tmp=eeg(:,1:sampl);
    traindata=traindata1(:,:,1:sampl,:);
    model = train_trca(traindata, 250, 5);
    for fb_i = 1:1:model.num_fbs
        testdata = filterbank(test_tmp, model.fs, fb_i);
        for class_i = 1:1:model.num_targs
            traindata =  squeeze(model.trains(class_i, fb_i, :, :));
                w = squeeze(model.W(fb_i, class_i, :));
           %原始
            r_tmp = corrcoef(testdata'*w, traindata'*w);
            r(fb_i,class_i) = r_tmp(1,2);
         %   [~,~,r_tmp] = canoncorr(testdata'*w, traindata'*w);
         %    r(fb_i, class_i) = r_tmp(1,1);
        end % class_i
    end % fb_i
     rho = fb_coefs*r;
     rho=softmax(rho')';
      for k=1:1:40
         if k==1
             b(temp)=rho(k);
         else
             b((k-1)*70+temp)=rho(k);
         end
     end
     temp=temp+1;
     
    for j=1:1:40
        aa((j-1)*70+e)=ttt;    
    end
    %提取时间
    sj=aa(1,[1:e 71:70+e 141:140+e 211:210+e 281:280+e 351:350+e 421:420+e 491:490+e 561:560+e 631:630+e 701:700+e 771:770+e 841:840+e 911:910+e 981:980+e 1051:1050+e 1121:1120+e 1191:1190+e 1261:1260+e 1331:1330+e 1401:1400+e 1471:1470+e 1541:1540+e 1611:1610+e 1681:1680+e 1751:1750+e 1821:1820+e 1891:1890+e 1961:1960+e 2031:2030+e 2101:2100+e 2171:2170+e 2241:2240+e 2311:2310+e 2381:2380+e 2451:2450+e 2521:2520+e 2591:2590+e 2661:2660+e 2731:2730+e]);
    %提取特征
    tezheng=b(1,[1:e 71:70+e 141:140+e 211:210+e 281:280+e 351:350+e 421:420+e 491:490+e 561:560+e 631:630+e 701:700+e 771:770+e 841:840+e 911:910+e 981:980+e 1051:1050+e 1121:1120+e 1191:1190+e 1261:1260+e 1331:1330+e 1401:1400+e 1471:1470+e 1541:1540+e 1611:1610+e 1681:1680+e 1751:1750+e 1821:1820+e 1891:1890+e 1961:1960+e 2031:2030+e 2101:2100+e 2171:2170+e 2241:2240+e 2311:2310+e 2381:2380+e 2451:2450+e 2521:2520+e 2591:2590+e 2661:2660+e 2731:2730+e]);
    
    e=e+1;
    

[~,atab,ctab,stats] = aoctool(sj,tezheng,group,[],[],[],[],'off');
disp(cell2mat(atab(4,6)));
if cell2mat(atab(4,6))<0.05
slopez=ctab(44:83,2)';
slopez=cell2mat(slopez);
[~,slope]=max(slopez);
c=multcompare(stats);
aaa=c(1:780,1:2);
ccc=0;
for l=1:1:40
    if l<slope
        bbb=[l,slope];
        [~,ind]=ismember(bbb,aaa,'rows');
        if c(ind,6)>=0.05
               ttt=ttt+0.05;
               break;
        else
            ccc=ccc+1;
        end
    end
     if l>slope
            bbb=[slope,l];
            [~,ind]=ismember(bbb,aaa,'rows');
            if c(ind,6)>=0.05
                ttt=ttt+0.05;
                break;
            else
                ccc=ccc+1;
            end
     end
end
 if ccc==39
     break;
 end
else
    ttt=ttt+0.05;
end
end 


    [~, tau] = max(rho);
    results(targ_i) = tau;
    time(targ_i)=ttt;
 
    
   
end
    
    is_correct = (results==labels);
    accs(loocv_i) = mean(is_correct)*100;
    fprintf('Trial %d: Accuracy = %2.2f%% \n',...
        loocv_i, accs(loocv_i) );
    add1='C:\atrca\结果3\s';
    add2=num2str(loocv_i);
    add3=num2str(o);
    add4='block';
    add5='\';
    pathname=strcat(add1,add3,add5);
    filename=strcat(add4,add2);
    save([pathname,filename],'accs','results','time');
    
end
end
    %model 
    function model = train_trca(eeg, fs, num_fbs)


if nargin < 2
    error('stats:train_trca:LackOfInput', 'Not enough input arguments.'); 
end

if ~exist('num_fbs', 'var') || isempty(num_fbs), num_fbs = 3; end

[num_targs, num_chans, num_smpls, ~] = size(eeg);
trains = zeros(num_targs, num_fbs, num_chans, num_smpls);
W = zeros(num_fbs, num_targs, num_chans);
for targ_i = 1:1:num_targs
    eeg_tmp = squeeze(eeg(targ_i, :, :, :));
    for fb_i = 1:1:num_fbs
        eeg_tmp = filterbank(eeg_tmp, fs, fb_i);
        trains(targ_i,fb_i,:,:) = squeeze(mean(eeg_tmp, 3));
        w_tmp = trca(eeg_tmp);
        W(fb_i, targ_i, :) = w_tmp(:,1);
    end % fb_i
end % targ_i
model = struct('trains', trains, 'W', W,...
    'num_fbs', num_fbs, 'fs', fs, 'num_targs', num_targs);


function W = trca(eeg)


[num_chans, num_smpls, num_trials]  = size(eeg);
S = zeros(num_chans);
for trial_i = 1:1:num_trials-1
    x1 = squeeze(eeg(:,:,trial_i));
    x1 = bsxfun(@minus, x1, mean(x1,2));
    for trial_j = trial_i+1:1:num_trials
        x2 = squeeze(eeg(:,:,trial_j));
        x2 = bsxfun(@minus, x2, mean(x2,2));
        S = S + x1*x2' + x2*x1';
    end % trial_j
end % trial_i
UX = reshape(eeg, num_chans, num_smpls*num_trials);
UX = bsxfun(@minus, UX, mean(UX,2));
Q = UX*UX';
[W,~] = eigs(S, Q);
end
    end
 