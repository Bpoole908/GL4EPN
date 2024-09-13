% Set MATPOWER case and options
power_case = 'case14';
mpopt = mpoption('out.all', 0, 'verbose', 0);
% Load MATPOWER case to extract information for args
mpc = loadcase(power_case);

% Build voltage dataset
rng(0) % Set seed
n_buses = height(mpc.bus);
features = 2;
samples = 5000;
n_columns = n_buses*features;
volt_data = zeros(samples, n_columns);
% Build column names
columns = strings(n_buses, features);
for i = 1:n_buses
    mag = sprintf("BUS%d_01_V_mag",i);
    ang = sprintf("BUS%d_01_V_ang",i);
    columns(i, :) = [mag, ang];
end
columns = [columns(:, 1).';columns(:, 2).'];
columns = columns(:);

% Build random load scaling
bounds = [{.8;.9},{.8;1.2}, {1.0; 1.1}, {1.1; 1.2}];

for b=bounds
    [b_min, b_max] = b{:};
    curr_date = string(datetime('today', 'Format', 'yyyy-MM-dd'));
    save_dir = sprintf('%s/%s/%0.3g-%0.3g', curr_date, power_case, b_min*100, b_max*100);
    save_path = sprintf("%s/Data_pu_Complete.csv",save_dir);
    if exist(save_path, 'file')
        fprintf("File %s already exists, continuing...\n",save_path)
        continue
    end

    load_scales = (b_max-b_min).*rand(samples,1) + b_min;
    load_scales = sort(load_scales);
    % load_scales(1) = 1;
    
    % Generate synthetic data
    for i = 1 : height(load_scales)
        mpc = loadcase(power_case);
        mpc = scale_load(load_scales(i), mpc);
        results = runopf(mpc, mpopt);
        volts = results.bus(:, 8:9);
        mag = volts(:, 1).';
        ang = volts(:, 2).';
        ang = deg2rad(ang);
        pmu = [mag;ang];
        volt_data(i, :) = pmu(:);
    
    end
    
    % Format and save data
    volt_table = array2table(cat(2, load_scales, volt_data), 'VariableNames', cat(1, 'Load Scales', columns));
    mkdir(save_dir);
    writetable(volt_table,  save_path, 'Delimiter',',');
end