# SoccerMomentToGif

SoccerMomentToGIF

SoccerMomentGIF is an innovative project that automatically creates GIFs capturing important moments from soccer games by analyzing the audio of the game. The script utilizes audio processing and speech recognition to identify significant events and then generates a GIF highlighting those moments.

  

## Features

*	Extracts audio from video files

*	Processes audio to identify key moments

*	Transcribes audio to identify important words and phrases

*	Merges audio-based and speech-based event lists

*	Generates a GIF showcasing the important moments

## Dependencies

*	tkinter

*	librosa

*	numpy

*	matplotlib

*	moviepy

*	PyQt5

*	SpeechRecognition

*	pydub

## How to use

1.	Make sure you have Python 3 and all dependencies installed.
2.	Run the script python soccer_moment_gif.py.
3.	The script will open a file dialog for you to select a video file (supported formats: .mp4, .avi, .mkv, .flv, .mov, .wmv).
4.	Once you've selected a video, the script will process the audio, transcribe it, and identify important moments.

5.	A GIF showcasing the important moments will be generated and saved in the same directory as the input video file.

## Code overview

The script is organized into several functions, each responsible for a specific task:

  

*	main(): The main function that ties all other functions together.

*	select_video_file(): Opens a file dialog for the user to select a video file.

*	calc_rms_and_rms_normalized(audio): Calculates the root-mean-square (RMS) energy and normalizes it.

*	plot_rms_energy(...): Plots the normalized RMS energy with the important moments highlighted.

*	find_important_parts(...): Identifies important moments in the audio based on RMS energy and a threshold.

*	calculate_mad_threshold(...): Calculates a threshold based on the median and mean absolute deviation of the RMS energy.

*	create_gif_from_intervals(...): Generates a GIF showcasing the important moments.

*	seconds_to_mmss_string(seconds): Converts seconds to a formatted MM:SS string.

*	getVideoAudio(video_file_path): Extracts audio from a video file.

*	choose_video(): Creates a GUI window for selecting the video file.

*	load_audio_file(file_path): Loads the audio file using Librosa.

*	important_words_list(): Returns a list of important words to be identified in the audio transcription.

*	transcribe_audio_file_to_console(...): Transcribes the audio file using Google Speech Recognition and identifies important timestamps.

*	intervals_to_time_strings(intervals): Converts a list of time intervals in seconds to a list of time strings in MM:SS format.
