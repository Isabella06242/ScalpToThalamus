clear
clc

seizureFiles = readtable('C:\Users\CashLab\Desktop\seizureList_withThalamicSpread.xlsx', ...
    'sheet', 'Sheet1');
seizureFiles = seizureFiles(1:308,1:5);

data_path = '\\172.17.146.246\Tarzan3\Seizures\MAT files converted from EDF\';
saving_path = '\\172.17.146.246\Tarzan3\Personnel_Folders\Yujie\trimmedEEG_SpreadToThalamus\';

for sz = 1:size(seizureFiles,1)

    peri_sz_time = [30 30];

    patient = cell2mat(seizureFiles{sz,1});
    sz_num = cell2mat(seizureFiles{sz,2});
    seizure_file_name = cell2mat(seizureFiles{sz,3});
    sz_onset_time = seizureFiles{sz,4};

    fprintf(['Running ' patient ',' sz_num '...\n']);

    seizure_path = [data_path,'\',patient,'\',seizure_file_name,'.mat'];
    load(seizure_path);
    rawdata = rawdata_edf;


    %% selecting the channels of interest (scalp channels) using Fieldtrip toolbox function
    scalp_elecs = {'FP1','F7','T3','T5','O1','F3','C3','P3',...
        'FP2','F8','T4','T6','O2','F4','C4','P4',...
        'FZ','CZ','PZ','T1','T2','Fz','Cz','Pz','Fp1','Fp2'};

    cfg         = [];
    cfg.channel = scalp_elecs;
    rawdata     = ft_selectdata(cfg, rawdata);

    %%
    %selecting data around the seizure

    [~, data_st_sample] = min(abs((sz_onset_time - peri_sz_time(1))-rawdata.time{1,1}));
    [~, data_end_sample] = min(abs((sz_onset_time + peri_sz_time(2)) -rawdata.time{1,1}));


    %%% Here the data with the current channels is again saved in a
    %%% Fieldtrip format. You could save it in any format that works for
    %%% your analysis
    data.fsample = rawdata.fsample;
    data.label = rawdata.label;
    data.trial{1,1} = rawdata.trial{1,1}(:,data_st_sample:data_end_sample);
    data.time{1,1} = 1/data.fsample:1/data.fsample:size(data.trial{1,1},2)/data.fsample;

    %%% Save one clip per seizure. Name is keyed on sz_num (unique per row)
    %%% so multiple seizures sharing one EDF file (e.g. MG90b_Sz5-7 -> Sz5,
    %%% Sz6, Sz7) no longer overwrite each other. The file name itself
    %%% contains no folder path; the output folder is given to save() via
    %%% fullfile, so it is not baked into the file name.
    saved_file_name = [patient '_' sz_num '_scalp'];
    save(fullfile(saving_path, saved_file_name), 'data', '-v7.3');

    clear data    % avoid leaking fields between seizures

end

