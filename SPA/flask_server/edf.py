import datetime
import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from mne.time_frequency import psd_array_multitaper
from scipy.stats import linregress
from scipy.signal import hilbert
import re
import yasa
"""unified interface: make the output one pandas.DataFrame (if possible)
fix previous problems:
stratify by stage: W, N1, N2, N3, REM, NREM
[optional, I forget to mention during meeting] we should exclude times before lights off and after lights on (need time from annotation); W can be further divided into wake before sleep onset, wake after sleep onset, wake after sleep in the morning
computing slope
time axis: keep gap
time axis: unit in hours
Linear regression: use scipy.stats.linregress
install Luna for spindle detection: https://zzz.bwh.harvard.edu/luna/download/
Install LunaC and LunaR
Install Luna python package `lunapi`
[Luna is difficult to install... a bit easier under Linux. If you can't manage to install it. you can switch to YASA]
run the Luna/YASA tutorial about spindle detection
"""
plt.rcParams.update({'font.size': 16})

class EEGFeatureComputation:
    def __init__(self, edf_path, channels, stagepath, output_dir):
        self.edf_path = edf_path
        self.channels = channels
        self.stagepath = stagepath
        self.output_dir = output_dir
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'sigma': (11, 16),
            'beta': (12, 30),
            'slow_oscillation': (0.3, 1)
        }
        self.channels_dict = {
            'central': ['C3-M2', 'C4-M1'],
            'frontal': ['F3-M2', 'F4-M1'],
            'occipital': ['O1-M2', 'O2-M1']
        }
        self.psds = None
        self.freqs = None
        self.epochs = None
        self.start_time = None
        self.stages = self.load_stages()

    def load_stages(self):
        df = pd.read_csv(self.stagepath)
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
        sleep_stage_df = df[df['event'].str.contains('Sleep_stage_')]
        sleep_stage_df = sleep_stage_df.groupby('epoch').first().reset_index()
        sleep_stage_df['stage'] = sleep_stage_df['event'].str.replace('Sleep_stage_', '')
        return sleep_stage_df

    def filter_stages(self, results_df):
        # Filtering stages to exclude times before lights off and after lights on
        df = pd.read_csv(self.stagepath)
        pattern = r'(?i)(lights?\s*(?:out?|0|on|1))' #regex to capture lightsout/on
        matches = df['event'].str.extract(pattern, expand=False)
        match_indices = matches.dropna().index
        start_index = match_indices[0] if match_indices[0] != None else None
        end_index = match_indices[1] if match_indices[1] != None else None
        filtered_df = results_df.iloc[start_index + 1:end_index].reset_index(drop=True)
        return filtered_df

    def preprocess_data(self):
        raw = mne.io.read_raw_edf(self.edf_path, preload=False)
        raw = mne.io.read_raw_edf(self.edf_path, preload=True, exclude=[x for x in raw.ch_names if x not in self.channels])
        raw.notch_filter(freqs=60)
        raw.filter(l_freq=0.3, h_freq=35)
        
        sfreq = raw.info['sfreq']
        events = mne.make_fixed_length_events(raw, id=1, duration=30)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=30 - 1/raw.info['sfreq'], baseline=None, preload=True)
        
        epochs_data = epochs.get_data(picks=self.channels) * 1e6
        self.psds, self.freqs = psd_array_multitaper(epochs_data, sfreq=sfreq, fmin=0.3, fmax=35, n_jobs=1, bandwidth=0.5)
        self.psds = 10 * np.log10(self.psds)
        self.epochs = epochs
        self.start_time = raw.info['meas_date']

    def avg(self, psds, channel_names, channels):
        indices = [channel_names.index(ch) for ch in channels if ch in channel_names]
        return np.nanmean(psds[:, indices, :], axis=1)

    def get_band_power(self, psds_db, freqs, band_mask):
        psds = np.power(10, psds_db / 10)
        dfreq = freqs[1] - freqs[0]
        bp = np.nansum(psds[..., band_mask], axis=-1) * dfreq
        return 10 * np.log10(bp)

    def compute_band_powers(self):
        band_powers = {}
        for band, (fmin, fmax) in self.bands.items():
            band_mask = (self.freqs >= fmin) & (self.freqs <= fmax)
            band_powers[band] = {}
            for region, channels in self.channels_dict.items():
                avg_psds = self.avg(self.psds, self.epochs.ch_names, channels)
                band_powers[band][region] = self.get_band_power(avg_psds, self.freqs, band_mask)
        
        return band_powers
    
    def spect(self, psds, freqs, title, epoch_length, start_time, output_file):
        plt.close()
        plt.figure(figsize=(14, 5))

        # Average the PSDs across channels for each epoch
        avg_psds = np.nanmean(psds, axis=1)

        T = avg_psds.shape[0] * epoch_length / 3600
        plt.imshow(avg_psds.T, aspect='auto', origin='lower', extent=[0, T, freqs[0], freqs[-1]], cmap='turbo', vmin=15, vmax=30)

        xticks = np.arange(int(T) + 1)
        xticklabels = [(start_time + datetime.timedelta(hours=int(x))).strftime('%H:%M') for x in xticks]

        plt.colorbar(label='Power Spectral Density (dB/Hz)')
        plt.title(title)
        plt.xticks(xticks, labels=xticklabels)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (hours)')
        plt.savefig(output_file, dpi=500, bbox_inches='tight')

    def compute_slopes(self, band_powers):
        # Computing slopes for each band and region, stratified by sleep stage
        slopes = {}
        for band in self.bands.keys():
            slopes[band] = {}
            for region in self.channels_dict.keys():
                slopes[band][region] = {}
                for stage in ['W', 'N1', 'N2', 'N3', 'REM', 'NREM', 'R']:
                    # Stratify by stage, including NREM (N1+N2+N3) and R (REM), REM = REM, W = Wake
                    if stage == 'NREM':
                        stage_mask = self.stages['stage'].isin(['N1', 'N2', 'N3'])
                    elif stage == 'R':
                        stage_mask = self.stages['stage'] == 'REM'
                    else:
                        stage_mask = self.stages['stage'] == stage
                    
                    if np.sum(stage_mask) > 1:
                        # Calculate the cumulative time in hours with gaps maintained
                        times = np.arange(len(stage_mask)) * 30 / 3600  # Convert 30-second epochs to hours
                        cumulative_time = np.cumsum(stage_mask * 30 / 3600)
                        
                        y = band_powers[band][region][stage_mask]
                        # Use scipy.stats.linregress for linear regression
                        slope, _, _, _, _ = linregress(cumulative_time[stage_mask], y)
                        slopes[band][region][stage] = slope
                    else:
                        slopes[band][region][stage] = np.nan
        
        return slopes

    def create_unified_dataframe(self, band_powers, slopes):
        # Creating a unified DataFrame containing all computed features - final pandas
        results = []
        for epoch in range(len(self.epochs)):
            row = {
                'epoch': epoch + 1,
                'time': (self.start_time + datetime.timedelta(seconds=30*epoch)).strftime('%H:%M:%S')
            }
            stage = self.stages.loc[self.stages['epoch'] == epoch + 1, 'stage']
            if stage.empty:
                row['stage'] = 'Unknown'
            else:
                row['stage'] = stage.values[0]
            
            for band in self.bands.keys():
                for region in self.channels_dict.keys():
                    row[f'{band}_{region}_power'] = band_powers[band][region][epoch]
                    
                    if row['stage'] in ['NREM', 'W', 'N1', 'N2', 'N3', 'REM', 'R']:
                        row[f'{band}_{region}_slope_{row["stage"]}'] = slopes[band][region][row['stage']]
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def detect_spindles(self, data, sfreq):
        """
        Detect sleep spindles using YASA and integrate results with existing data.
        
        Parameters:
        data (np.ndarray): Numpy array of shape (n_channels, n_epochs, n_times)
        sfreq (float): Sampling frequency of the EEG data
        
        Returns:
        pd.DataFrame: DataFrame containing detected spindles integrated with existing data
        """
        n_epochs, n_channels, n_times = data.shape
        spindles_df = pd.DataFrame(index=range(1, n_epochs + 1))
        spindles_df['epoch'] = spindles_df.index
        
        data_reshaped = data.reshape(n_channels, -1)
        
        # Scale data to microvolts - yasa required it in this manner
        data_uv = data_reshaped * 1e6
        
        for ch_idx in range(n_channels):
            ch_name = f'channel_{ch_idx}' 
            
            sp = yasa.spindles_detect(data_uv[ch_idx], sfreq) #default parameters - dropdown list - basically parameter input
            
            if sp is not None:
                summary = sp.summary()
                if not summary.empty:
                    summary['Start'] = pd.to_timedelta(summary['Start'], unit='s')
                    summary['Start_time'] = self.start_time + summary['Start']
                    summary['epoch'] = (summary['Start'].dt.total_seconds() / 30).astype(int) + 1
                    x=1
                    agg_dict = {}
                    for col in ['Duration', 'Amplitude', 'Frequency']:
                        if col in summary.columns:
                            agg_dict[col] = 'mean'
                        else:
                            print(f"Warning: Column '{col}' not found in spindle summary for {ch_name}")
                    
                    agg_dict['Oscillations'] = 'size' #presume this is the number of spindles in a given epoch? but not sure
                    
                    grouped = summary.groupby('epoch').agg(agg_dict).reset_index()
                    
                    grouped.columns = [f'{col}_{ch_name}' if col != 'epoch' else col for col in grouped.columns]
                    
                    spindles_df = pd.merge(spindles_df, grouped, on='epoch', how='left')
                else:
                    print(f"No spindles detected for channel {ch_idx}")
            else:
                print(f"Spindle detection failed for channel {ch_idx}")
        
        spindles_df = spindles_df.fillna(0)
        
        results_df = pd.merge(self.create_unified_dataframe(self.compute_band_powers(), self.compute_slopes(self.compute_band_powers())), 
                            spindles_df, on='epoch', how='left') #all merging happens here; final output is pandasdf
        
        return results_df

    def detect_slow_oscillations(self, data, sfreq):
        """
        Detect slow oscillations using YASA and integrate results with existing data.
        
        Parameters:
        data (np.ndarray): Numpy array of shape (n_epochs, n_channels, n_times)
        sfreq (float): Sampling frequency of the EEG data
        
        Returns:
        pd.DataFrame: DataFrame containing detected slow oscillations integrated with existing data
        """
        n_epochs, n_channels, n_times = data.shape
        so_df = pd.DataFrame(index=range(1, n_epochs + 1))
        so_df['epoch'] = so_df.index
        
        data_reshaped = data.reshape(n_epochs * n_channels, n_times)
        
        # Scale data to microvolts
        data_uv = data_reshaped * 1e6
        
        for ch_idx in range(n_channels):
            ch_name = f'channel_{ch_idx}'
            epoch_sos = []
            
            for epoch_idx in range(n_epochs):
                epoch_data = data_uv[epoch_idx * n_channels + ch_idx]
                
                so = yasa.sw_detect(epoch_data, sfreq)
                
                if so is not None:
                    summary = so.summary()
                    if not summary.empty:
                        summary['Start'] = pd.to_timedelta(summary['Start'], unit='s')
                        summary['Start_time'] = self.start_time + summary['Start']
                        summary['epoch'] = epoch_idx + 1
                        
                        epoch_sos.append(summary)
                    else:
                        print(f"No slow oscillations detected for epoch {epoch_idx}, channel {ch_idx}")
                else:
                    print(f"Slow oscillation detection failed for epoch {epoch_idx}, channel {ch_idx}")
            
            if epoch_sos:
                epoch_sos_df = pd.concat(epoch_sos)
                agg_dict = {
                    'Duration': 'mean',
                    'Start_time': 'mean',
                    'Frequency': 'mean',
                    'NegPeak': 'mean',
                    'MidCrossing': 'mean',
                    'PosPeak': 'mean'
                }
                
                grouped = epoch_sos_df.groupby('epoch').agg(agg_dict).reset_index()
                
                grouped.columns = [f'SO_{col}_{ch_name}' if col != 'epoch' else col for col in grouped.columns]
                
                so_df = pd.merge(so_df, grouped, on='epoch', how='left')
        
        so_df = so_df.fillna(0)
        return so_df

    def compute_spindle_so_coupling(self, data, sfreq):
        """
        Detect spindles and slow oscillations and compute their coupling using YASA.
        Parameters:
        data (np.ndarray): Numpy array of shape (n_epochs, n_channels, n_times)
        sfreq (float): Sampling frequency of the EEG data
        Returns:
        pd.DataFrame: DataFrame containing spindle-SO coupling metrics
        """
        
        coupling_results = pd.DataFrame(index=range(1, data.shape[0] + 1))
        coupling_results['epoch'] = coupling_results.index
        
        for ch_idx in range(data.shape[1]):
            ch_name = self.channels[ch_idx]
            channel_data = data[:, ch_idx, :].reshape(-1) * 1e6  # Convert to µV
            
            hypno = self.stages['stage'].map({'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}).fillna(-1).astype(int).values
            
            hypno = yasa.hypno_upsample_to_data(hypno=hypno, sf_hypno=1/30, data=channel_data, sf_data=sfreq)
            
            sw = yasa.sw_detect(channel_data, sfreq, ch_names=[ch_name], hypno=hypno, include=(2, 3), coupling=True,
                                coupling_params=dict(freq_sp=(12, 16), time=2, p=None))
            
            if sw is not None:
                events = sw.summary()
                if not events.empty:
                    events['epoch'] = (events['Start'] / 30).astype(int) + 1
                    
                    numeric_columns = events.select_dtypes(include=[np.number]).columns.tolist()
                    if 'epoch' in numeric_columns:
                        numeric_columns.remove('epoch')
                    
                    events_grouped = events.groupby('epoch')[numeric_columns].mean()
                    
                    events_grouped.columns = [f'{col}_{ch_name}' for col in events_grouped.columns]
                    
                    coupling_results = pd.merge(coupling_results, events_grouped, left_on='epoch', right_index=True, how='left')
        
        return coupling_results.fillna(0)




    def run(self):
        self.preprocess_data()
        band_powers = self.compute_band_powers()
        slopes = self.compute_slopes(band_powers)
        
        results_df = self.detect_spindles(self.epochs.get_data(), self.epochs.info['sfreq'])
        so_df = self.detect_slow_oscillations(self.epochs.get_data(), self.epochs.info['sfreq'])
        coupling_df = self.compute_spindle_so_coupling(self.epochs.get_data(), self.epochs.info['sfreq'])
        
        # Merge all results
        results_df = pd.merge(results_df, so_df, on='epoch', how='left')
        results_df = pd.merge(results_df, coupling_df, on='epoch', how='left')
        
        filtered_results_df = self.filter_stages(results_df)
        print("Filtered results are saved.")

        spectrogram_output = os.path.join(self.output_dir, 'spectrogram.png')
        self.spect(self.psds, self.freqs, 'EEG Spectrogram', 30, self.start_time, spectrogram_output)
        print("Spectrogram saved.")

        return filtered_results_df



if __name__ == "__main__":
    edf_path = "sample_psg.edf"
    stagepath = "sample_psg_annotations.csv"
    output_dir = r"C:\Users\sendm\Desktop\Programming\Code\Python\Python\SPA\flask_server\images"
    channels = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']
    
    eeg_feature_computation = EEGFeatureComputation(edf_path, channels, stagepath, output_dir)
    results_df = eeg_feature_computation.run()
    results_df.to_csv(os.path.join(output_dir, 'filtered_eeg_features1.csv'), index=False)
    x=1
    print(results_df.head())