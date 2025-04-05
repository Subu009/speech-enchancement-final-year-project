import argparse

parser = argparse.ArgumentParser(description='Speech enhancement,data creation, training and prediction')

parser.add_argument('--mode',default='prediction', type=str, choices=['data_creation', 'training', 'prediction'])

parser.add_argument('--noise_dir', default='/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/noise', type=str)

parser.add_argument('--voice_dir', default='/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/clean_voice', type=str)

parser.add_argument('--path_save_spectrogram', default='/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/spectrogram/', type=str)

parser.add_argument('--path_save_time_serie', default='/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/time_serie/', type=str)

parser.add_argument('--path_save_sound', default='/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/sound/', type=str)

parser.add_argument('--nb_samples', default=50, type=int)

parser.add_argument('--training_from_scratch',default=True, type=bool)

parser.add_argument('--weights_folder', default='./weights', type=str)

parser.add_argument('--epochs', default=10, type=int)

parser.add_argument('--batch_size', default=20, type=int)

parser.add_argument('--name_model', default='model_unet', type=str)

parser.add_argument('--audio_dir_prediction', default='./demo_data/test', type=str)

parser.add_argument('--dir_save_prediction', default='./demo_data/save_predictions/', type=str)

parser.add_argument('--audio_input_prediction', default=['noisy_voice_long_t2.wav'], type=list)

parser.add_argument('--audio_output_prediction', default='denoise_t2.wav', type=str)

parser.add_argument('--sample_rate', default=8000, type=int)

parser.add_argument('--min_duration', default=1.0, type=float)

parser.add_argument('--frame_length', default=8064, type=int)

parser.add_argument('--hop_length_frame', default=8064, type=int)

parser.add_argument('--hop_length_frame_noise', default=5000, type=int)

parser.add_argument('--n_fft', default=255, type=int)

parser.add_argument('--hop_length_fft', default=63, type=int)
