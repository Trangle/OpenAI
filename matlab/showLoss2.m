clc;close all;clear;

basepath_base = 'D:\MyCoding\DeepLearning\test_example\mxnet\example\kesci_bot_animal\Logs';
% basepath_2 = 'D:\MXNet\image-classification\checkpoints\cifar10-resnet+inception20';
% basepath_3 = 'D:\MXNet\image-classification\checkpoints\cifar10-resnet-mybase';


% basepaths ={basepath_base,basepath_2,basepath_3};
basepaths = {basepath_base};

N = length(basepaths);
TrainAccu = cell(N,1);
TestAccu = cell(N,1);
for k=1:N
    L = dir([basepaths{k} '\*.txt']);
    logPath = cell(length(L),1);
    for j=1:length(L)
        logPath{j} = [basepaths{k} '\' L(j).name];
    end
    [TrainAccu{k},TestAccu{k}] = ReadLoss( logPath );
end

colors={[1 0 0],[0 1 0],[0 0 1],[0.5 0 0.5],[1 0.5 0],[0,0.25,0]};
figure(1),
hold on
for k=1:N
plot((1:length(TrainAccu{k}))/2,100*(1-TrainAccu{k}),'-.','Color',colors{k});
plot((1:length(TestAccu{k})),100*(1-TestAccu{k}),'-','Color',colors{k});
end

ylim([0,20]);
title('cifar10 (NSize=3)')
xlabel('epoch');
ylabel('Top-1 error(%)');
grid on
legend('base-train','base-test',...
'res+inp-train','res+inp-test')



