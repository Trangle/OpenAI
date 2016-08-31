function [TrainAcc,TestAcc ] = ReadLoss( filenames )

    %% Format string for each line of text:
    formatSpec = '%s%[^\n\r]';
    delimiter = '';

    TrainAcc = [];
    TestAcc = [];
    
    for f=1:length(filenames)
        filename = filenames{f};
        fileID = fopen(filename,'r');
        dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true,  'ReturnOnError', false);
        fclose(fileID);
        log = [dataArray{1:end-1}];

        N = length(log);
        iter = 0;
        for k=1:N
            line = log{k};
       
%             idx = strfind(line,'Epoch[');
%             if ~isempty( idx )
%                 [res,count,errmsg] = sscanf(line(idx:end),'Epoch[%d] Batch [%d]');
% %                 iter = round( res[1]*num_iters_per_epoch + res[2] );
%             end
            
            idx = strfind(line,'Train-accuracy=');
            if ~isempty( idx )
                [acc,count,errmsg] = sscanf(line(idx:end),'Train-accuracy=%f');
                TrainAcc = [TrainAcc acc];
            end

            idx = strfind(line,'Validation-accuracy=');
            if ~isempty( idx )
                [acc,count,errmsg] = sscanf(line(idx:end),'Validation-accuracy=%f');
                TestAcc = [TestAcc acc];
            end   
        end
    end
    
%     TrainAcc = TrainAcc(1:max_iter-1);
%     TestAcc = TestAcc(1:round(max_iter/15)-1);
    
%     LFWAcc =100- LFWAcc;

end

