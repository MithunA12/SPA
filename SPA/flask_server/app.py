from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import mne
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import subprocess
import matplotlib.ticker as ticker
import matplotlib

matplotlib.use('Agg')  # non-interactive backend for matplotlib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'images'
ANNOTATION_FOLDER = 'annotations'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['ANNOTATION_FOLDER'] = ANNOTATION_FOLDER

if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

if not os.path.exists(ANNOTATION_FOLDER):
    os.makedirs(ANNOTATION_FOLDER)

def clear_image_directory():
    """Delete all files in the image directory."""
    for filename in os.listdir(app.config['IMAGE_FOLDER']):
        file_path = os.path.join(app.config['IMAGE_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  #
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) 
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and return channel names."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    raw_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(raw_file_path)

    annotation_filename = ''
    if 'annotation_file' in request.files:
        annotation_file = request.files['annotation_file']
        annotation_filename = secure_filename(annotation_file.filename)
        annotation_file_path = os.path.join(app.config['UPLOAD_FOLDER'], annotation_filename)
        annotation_file.save(annotation_file_path)

    raw = mne.io.read_raw_edf(raw_file_path, preload=True)
    channel_names = raw.ch_names

    return jsonify({'channelNames': channel_names, 'filename': filename, 'annotationFilename': annotation_filename})

@app.route('/images', methods=['GET'])
def list_images():
    image_files = [f for f in os.listdir(app.config['IMAGE_FOLDER']) if f.endswith('.png')]
    return jsonify({'images': sorted(image_files)})

@app.route('/generate_images', methods=['POST'])
def generate_images():
    """Generate images and spectrograms from the uploaded EDF file"""
    data = request.get_json()
    index = data.get('index', 0)
    filename = data.get('filename')
    channel_mappings = data.get('channelMappings', {})
    annotation_data = data.get('annotation', {})
    annotation_filename = data.get('annotationFilename', '')

    raw_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    sleep_stage_signal_resampled = None
    if annotation_filename:
        # Process the annotation file for sleep stages
        annotation_file_path = os.path.join(app.config['UPLOAD_FOLDER'], annotation_filename)
        annotations_df = pd.read_csv(annotation_file_path)

        # Extract sleep stage annotations
        sleep_stages = annotations_df[annotations_df['event'].str.startswith('Sleep_stage_')].copy()

        # Convert 'time' column to datetime
        sleep_stages['time'] = pd.to_datetime(sleep_stages['time'], format='%H:%M:%S').dt.time

        sleep_stage_mapping = {
            'Sleep_stage_W': 4,
            'Sleep_stage_N1': 3,
            'Sleep_stage_N2': 2,
            'Sleep_stage_N3': 1,
            'Sleep_stage_R': 0,
            'Sleep_stage_?': -1,
            'Arousal - RespEvent - 1': -1,
            'Position - Left - 1': -1,
            'Treatment - None - 1': -1,
            'LIGHTS OUT': -1,
            'PLM - Periodic - 1': -1,
            'RespEvent - RERA - 6': -1,
            'RespEvent - Hypopnea - 4': -1,
            'RespEvent - ObstructiveApnea - 1': -1,
            'Position - Right - 1': -1,
            'Position - Supine - 1': -1,
            'PLM - Isolated - 1': -1,
            'RespEvent - CentralApnea - 2': -1,
            'LIGHTS ON': -1
        }

        sleep_stages = sleep_stages.reset_index()
        sleep_stages = sleep_stages.rename(columns={'time': 'start', 'duration': 'end'})

        # Convert 'start' and 'end' times to seconds
        sleep_stages['start'] = sleep_stages['start'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)
        sleep_stages['end'] = sleep_stages.apply(lambda row: row['start'] + row['end'], axis=1)
        sleep_stages['event'] = sleep_stages['event'].apply(lambda event: sleep_stage_mapping.get(event, -1))

        max_length = sleep_stages['end'].max()
        sleep_stage_signal = np.full(int(max_length), np.nan)

        # Fills the sleep_stage_signal array with sleep stage values
        for _, row in sleep_stages.iterrows():
            sleep_stage_signal[int(row['start']):int(row['end'])] = int(row['event'])

        sleep_stage_signal = np.nan_to_num(sleep_stage_signal, nan=-1)
        raw = mne.io.read_raw_edf(raw_file_path, preload=True)

        fs = raw.info['sfreq']
        total_duration = raw.n_times / fs
        sleep_stage_signal = sleep_stage_signal[:int(total_duration)]

        # Resample sleep stage signal
        sleep_stage_signal_resampled = np.repeat(sleep_stage_signal, 200)

    else:
        raw = mne.io.read_raw_edf(raw_file_path, preload=True)

    current_channel_names = raw.ch_names

    # Apply channel mappings
    new_channel_names = current_channel_names[:]
    for idx, name in channel_mappings.items():
        if int(idx) < len(new_channel_names):
            new_channel_names[int(idx)] = name

    rename_mapping = dict(zip(current_channel_names, new_channel_names))
    raw.rename_channels(rename_mapping)

    # Add annotations if provided
    onset = float(annotation_data.get('start', 0))
    duration = float(annotation_data.get('duration', 0))
    description = annotation_data.get('description', '')
    if onset and duration and description:
        annotations = mne.Annotations(onset=[onset], duration=[duration], description=[description])
        raw.set_annotations(annotations)

    # Generate EEG plots in groups of tens
    for i in range(index, index + 10):
        raw_fig = raw.plot(n_channels=21, start=i*90, duration=90, show_scrollbars=False, show=False, time_format='clock')
        raw_fig.savefig(f'{IMAGE_FOLDER}/raw_image_{i}.png')
        plt.close(raw_fig)

        # Generate sleep stage plots if available
        if sleep_stage_signal_resampled is not None:
            plt.figure(figsize=(15, 5))
            plt.plot(
                raw.times[i*90*200:(i+1)*90*200],
                sleep_stage_signal_resampled[i*90*200:(i+1)*90*200],
                label='Sleep Stages',
                drawstyle='steps-post'
            )
            plt.xticks([])
            plt.yticks(
                ticks=list(sleep_stage_mapping.values()),
                labels=[key.replace('Sleep_stage_', '') for key in sleep_stage_mapping.keys()]
            )
            plt.title('Sleep Stage Signal')
            plt.savefig(f'{IMAGE_FOLDER}/sleep_stage_image_{i}.png')
            plt.close()

    # Call script to generate spectrogram and CSV file with features
    edf_script_path = 'edf.py'
    subprocess.run(['python', edf_script_path, raw_file_path, annotation_file_path, IMAGE_FOLDER])

    new_images = [f'raw_image_{i}.png' for i in range(index, index + 10)] + [f'sleep_stage_image_{i}.png' for i in range(index, index + 10)]
    spectrogram_image = 'spectrogram.png'
    csv_file = 'filtered_eeg_features1.csv'
    return jsonify({'images': new_images, 'spectrogram': spectrogram_image, 'csv': csv_file})

@app.route('/images/<filename>')
def get_image(filename):
    """Serve image files"""
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    """Allow file download from the image directory"""
    return send_from_directory(app.config['IMAGE_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
