clc;
clear;
%load data from different deep networks
load whole_data1;
whole_data1 = whole_data;
whole_data1.train_fea = exp(whole_data1.train_fea);
whole_data1.test_fea = exp(whole_data1.test_fea);

temp = zeros(whole_data1.N_train,90);
for i = 1:whole_data1.N_train
    temp(i,:) = whole_data1.train_fea(i,:)/sum(whole_data1.train_fea(i,:));
end
whole_data1.train_fea = temp;

temp = zeros(whole_data1.N_test,90);
for i = 1:whole_data1.N_test
    temp(i,:) = whole_data1.test_fea(i,:)/sum(whole_data1.test_fea(i,:));
end
whole_data1.test_fea = temp;

load whole_data2;
whole_data2 = whole_data;
whole_data2.train_fea = exp(whole_data2.train_fea);
whole_data2.test_fea = exp(whole_data2.test_fea);

temp = zeros(whole_data2.N_train,90);
for i = 1:whole_data2.N_train
    temp(i,:) = whole_data2.train_fea(i,:)/sum(whole_data2.train_fea(i,:));
end
whole_data2.train_fea = temp;

temp = zeros(whole_data2.N_test,90);
for i = 1:whole_data2.N_test
    temp(i,:) = whole_data2.test_fea(i,:)/sum(whole_data2.test_fea(i,:));
end
whole_data2.test_fea = temp;

load whole_data3;
whole_data3 = whole_data;
whole_data3.train_fea = exp(whole_data3.train_fea);
whole_data3.test_fea = exp(whole_data3.test_fea);

temp = zeros(whole_data3.N_train,90);
for i = 1:whole_data3.N_train
    temp(i,:) = whole_data3.train_fea(i,:)/sum(whole_data3.train_fea(i,:));
end
whole_data3.train_fea = temp;

temp = zeros(whole_data3.N_test,90);
for i = 1:whole_data3.N_test
    temp(i,:) = whole_data3.test_fea(i,:)/sum(whole_data3.test_fea(i,:));
end
whole_data3.test_fea = temp;

load whole_data4;
whole_data4 = whole_data;
whole_data4.train_fea = exp(whole_data4.train_fea);
whole_data4.test_fea = exp(whole_data4.test_fea);

temp = zeros(whole_data4.N_train,90);
for i = 1:whole_data4.N_train
    temp(i,:) = whole_data4.train_fea(i,:)/sum(whole_data4.train_fea(i,:));
end
whole_data4.train_fea = temp;

temp = zeros(whole_data4.N_test,90);
for i = 1:whole_data4.N_test
    temp(i,:) = whole_data4.test_fea(i,:)/sum(whole_data4.test_fea(i,:));
end
whole_data4.test_fea = temp;



% preprocess data

N_train = whole_data1.N_train;
N_test = whole_data1.N_test;

train_fea = (whole_data1.train_fea + whole_data2.train_fea + whole_data3.train_fea + whole_data4.train_fea)/4;
test_fea = (whole_data1.test_fea + whole_data2.test_fea + whole_data3.test_fea + whole_data4.test_fea)/4;

train_label = whole_data1.train_label;

[~,test_label1] = max(whole_data1.test_fea,[],2);
[~,test_label2] = max(whole_data2.test_fea,[],2);
[~,test_label3] = max(whole_data3.test_fea,[],2);
[~,test_label4] = max(whole_data4.test_fea,[],2);
multi_test_label = [test_label1,test_label2,test_label3,test_label4];
label_std = std(multi_test_label,[],2);
index =  find(label_std>=6);


%essemble
dis = zeros(N_train,N_test);
for i = 1:N_train
    for j =1:N_test
        dis(i,j) = sum((train_fea(i,:)-test_fea(j,:)).^2);
    end
end

alpha = 1000;
K_temp = exp(-alpha*dis');
for i = 1:N_test
    K(i,:) = K_temp(i,:)/sum(K_temp(i,:));
end
test_label = K*train_label;
save result_final test_label;

fp = fopen('Predictions.csv','wt');
load('result_final.mat');
load('test_imdb.mat');
len = size(test_label,1);
for i =1 : len
    fprintf(fp, '%s,%d\n', test_imdb.images.name{i},test_label(i));
end
fclose(fp);

