clear
clc

folder = '\\172.17.146.246\Tarzan3\Personnel_Folders\Yujie\trimmedEEG_SpreadToThalamus\';
target_fs = 1024;

files = dir(fullfile(folder, '*.mat'));
if isempty(files)
    fprintf('No .mat files found in:\n  %s\n', folder);
    return
end

n_ok   = 0;
n_bad  = 0;
bad_files = {};

for k = 1:numel(files)
    fpath = fullfile(folder, files(k).name);

    try
        tmp = load(fpath, 'data');
        fs  = tmp.data.fsample;
    catch ME
        fprintf('[ERROR] %s — could not read fsample: %s\n', files(k).name, ME.message);
        n_bad = n_bad + 1;
        bad_files{end+1} = files(k).name; %#ok<AGROW>
        continue
    end

    if fs == target_fs
        fprintf('[OK]  %s — fsample = %g Hz\n', files(k).name, fs);
        n_ok = n_ok + 1;
    else
        fprintf('[BAD] %s — fsample = %g Hz (expected %g)\n', files(k).name, fs, target_fs);
        n_bad = n_bad + 1;
        bad_files{end+1} = files(k).name; %#ok<AGROW>
    end
end

fprintf('\n--- Summary ---\n');
fprintf('Total files : %d\n', numel(files));
fprintf('OK (%g Hz)  : %d\n', target_fs, n_ok);
fprintf('Not OK      : %d\n', n_bad);

if ~isempty(bad_files)
    fprintf('\nFiles that are NOT %g Hz:\n', target_fs);
    for k = 1:numel(bad_files)
        fprintf('  %s\n', bad_files{k});
    end
end
