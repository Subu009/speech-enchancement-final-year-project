from args import parser
import os
from preparedate import create_data
from train import training
from denoise import prediction

if __name__ == '__main__':

    args = parser.parse_args()

    mode = args.mode

    data_mode = False
    training_mode = False
    prediction_mode = False

    if mode == 'prediction':
        prediction_mode = True
    elif mode == 'training':
        training_mode = True
    elif mode == 'data_creation':
        data_mode = True

    if data_mode:

        noise_dir = args.noise_dir
        voice_dir = args.voice_dir
        path_save_time_serie = args.path_save_time_serie
        path_save_sound = args.path_save_sound
        path_save_spectrogram = args.path_save_spectrogram
        sample_rate = args.sample_rate
        min_duration = args.min_duration
         frame_length = args.frame_length
        hop_length_frame = args.hop_length_frame
        hop_length_frame_noise = args.hop_length_frame_noise
        nb_samples = args.nb_samples
        n_fft = args.n_fft
        hop_length_fft = args.hop_length_fft

        create_data(noise_dir, voice_dir, path_save_time_serie, path_save_sound, path_save_spectrogram, sample_rate,
        min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft)


    elif training_mode:
        path_save_spectrogram = args.path_save_spectrogram
        weights_path = args.weights_folder
        name_model = args.name_model
        training_from_scratch = args.training_from_scratch
        epochs = args.epochs
        batch_size = args.batch_size

        training(path_save_spectrogram, weights_path, name_model, training_from_scratch, epochs, batch_size)

    elif prediction_mode:
        weights_path = args.weights_folder
        name_model = args.name_model
        audio_dir_prediction = args.audio_dir_prediction
        dir_save_prediction = args.dir_save_prediction
        audio_input_prediction = args.audio_input_prediction
        audio_output_prediction = args.audio_output_prediction
        sample_rate = args.sample_rate
        min_duration = args.min_duration
        frame_length = args.frame_length
        hop_length_frame = args.hop_length_frame
        n_fft = args.n_fft
        hop_length_fft = args.hop_length_fft

        prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
        audio_output_prediction, sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft)
