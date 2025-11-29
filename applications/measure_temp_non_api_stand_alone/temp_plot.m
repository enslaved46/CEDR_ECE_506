% Author : Ashish 
% Credit to internet , got helped from internet

% Open file, no file handler, file must be presetn in the same path as the
% script
% comment and uncomment file as need

% fid = fopen('idle_lpd_measured_temp.txt', 'r');
% fid = fopen('idle_fpd_measured_temp.txt', 'r');

%fid = fopen('sar_100_fpd_measured_temp.txt', 'r');
 fid = fopen('sar_100_lpd_measured_temp.txt', 'r');

 %fid = fopen('sar_100_random_sch_lpd_measured_temp.txt', 'r');
 %fid = fopen('sar_100_random_sch_fpd_measured_temp.txt', 'r');
 
% fid = fopen('sar_100_HEFT_RT_fpd_measured_temp.txt' , 'r');
% fid = fopen('sar_100_HEFT_RT_lpd_measured_temp.txt' , 'r');

% Read as columns
% 1nd col: temperature (float)
% 2rd col: literal "C"
% 3th col: timestamp (uint64)
% 4th col: literal "ns"

data = textscan(fid, '%f C %f ns');
fclose(fid);

temp = data{1};   % temperature in C
t_ns = data{2};   % timestamp in ns (as double here)

% Convert time to seconds relative to first captured sample
% captured time is in ns since boot time
t_s = (t_ns - t_ns(1)) * 1e-9;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIlter 1 : oving avg filter(LPF)
% Removing small fluctuations in the sample
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure (1);
y = movmean(temp, 5);   % smooth over 5 samples
plot(t_s, temp); hold on;
plot(t_s, y,'LineWidth',2);
legend("Raw","Moving Average");
xlabel('Time (s)')
ylabel('Temperature (°C)');
grid on;

% saveas(gcf, 'fpd_moving_avg_simple_sch_100_runs.png');
saveas(gcf, 'lpd_moving_avg_simple_sch_100_runs.png');

% saveas(gcf, 'lpd_moving_avg_random_sch_100_runs.png');
% saveas(gcf, 'fpd_moving_avg_random_sch_100_runs.png');

% saveas(gcf, 'fpd_moving_avg_heft_rt_sch_100_runs.png');
% saveas(gcf, 'lpd_moving_avg_heft_rt_sch_100_runs.png');

% saveas(gcf, 'lpd_moving_avg_idle.png');
% saveas(gcf, 'fpd_moving_avg_idle.png');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filter 2 : LPF as butterworth
% Removes high-frequency noise, preserves slow trend.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2)
[b,a] = butter(3, 0.1);     % 3rd order, cutoff=0.1 Nyquist
y = filtfilt(b,a,temp);
plot(temp); hold on; plot(y,'LineWidth',2);
legend("Raw","LP Filtered");
xlabel('Samples')
ylabel('Temperature (°C)')
grid on;

% saveas(gcf, 'fpd_lpf_simple_sch_100_runs.png')
saveas(gcf, 'lpd_lpf_simple_sch_100_runs.png')

% saveas(gcf, 'lpd_lpf_random_sch_100_runs.png');
% saveas(gcf, 'fpd_lpf_random_sch_100_runs.png')

% saveas(gcf, 'fpd_lpf_heft_rt_sch_100_runs.png')
% saveas(gcf, 'lpd_lpf_heft_rt_sch_100_runs.png')

% saveas(gcf, 'lpd_lpf_idle.png')
% saveas(gcf, 'fpd_lpf_idle.png')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filter 3 : Median filter
% removing spikes random
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(3);
y = medfilt1(temp, 5); % window length = 5
plot(temp); hold on; plot(y);
grid on;
xlabel('Samples')
ylabel('Temperature (°C)')
title('Median Filtered Temperature')
legend('Raw Temperature','Median Filtered','Location','best')

% saveas(gcf, 'fpd_median_simple_sch_100_runs.png');
saveas(gcf, 'lpd_median_simple_sch_100_runs.png');

% saveas(gcf, 'lpd_median_random_sch_100_runs.png');
% saveas(gcf, 'fpd_median_random_sch_100_runs.png');

 %saveas(gcf, 'fpd_median_heft_rt_sch_100_runs.png');
 % saveas(gcf, 'lpd_median_heft_rt_sch_100_runs.png');

% saveas(gcf, 'lpd_median_idle.png');
%saveas(gcf, 'fpd_median_idle.png');


%HPF
% Keeping fluctuation
%figure(4);
%[b,a] = butter(3, 0.02,'high');
%y = filtfilt(b,a,temp);
%plot(t_s, temp, '-o'); hold on;
%plot(t_s, y, 'LineWidth', 2);
%grid on;

%legend('Raw Temperature','Filtered (High-Pass)');
%xlabel('Time (s)');
%ylabel('Temperature (°C)');
%title('High-Pass Filtered Temperature Signal');