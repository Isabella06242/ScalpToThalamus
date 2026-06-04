% Trim EEG rawdata to 30s before seizure onset and 30s after seizure end
% Data: 276 channels x 7351040 samples, fs = 1024 Hz
% Seizure window: 3662.578s (onset) to 3805.41s (end)

load('your_rawdata.mat');   % loads rawdata with fields: trial, time, label, fsample

fs          = rawdata.fsample;          % 1024
sz_onset    = 3662.578;                 % seconds
sz_end      = 3805.41;                  % seconds
peri_sz     = 30;                       % seconds padding each side

t_start     = sz_onset - peri_sz;       % 3632.578 s
t_end       = sz_end   + peri_sz;       % 3835.41  s

time_vec    = rawdata.time{1,1};

[~, idx_start] = min(abs(time_vec - t_start));
[~, idx_end]   = min(abs(time_vec - t_end));

trimmed             = rawdata;
trimmed.trial{1,1}  = rawdata.trial{1,1}(:, idx_start:idx_end);
trimmed.time{1,1}   = rawdata.time{1,1}(idx_start:idx_end);

fprintf('Trimmed data: %d channels x %d samples (%.2f s)\n', ...
    size(trimmed.trial{1,1}, 1), ...
    size(trimmed.trial{1,1}, 2), ...
    size(trimmed.trial{1,1}, 2) / fs);

% save('trimmed_seizure.mat', 'trimmed', '-v7.3');
