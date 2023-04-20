import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from moviepy.editor import *
from moviepy.editor import VideoFileClip
from PyQt5.QtWidgets import QApplication, QFileDialog
import speech_recognition as sr
import os
import time
from pydub import AudioSegment

import sys


def main():

    video_file_path = select_video_file()

    audio, audio_file_path = getVideoAudio(video_file_path)
    audio, sample_rate = load_audio_file(audio_file_path)
    rms, rms_normalized = calc_rms_and_rms_normalized(audio)
    file_name = os.path.basename(audio_file_path)
    multiplier=1.3
    threshold = calculate_mad_threshold(rms_normalized, multiplier=multiplier)
    min_interval_duration = 1.0

    important_parts_from_audio = find_important_parts(sample_rate, rms, rms_normalized, threshold, min_interval_duration)
    #print(f"Audio Intervals: {intervals_to_time_strings(important_parts_from_audio)}\n")
    
    #important_parts_from_text = transcribe_audio_file_to_console(audio_file_path, video_file_path)
    #print(f"Speech recognition Intervals: {intervals_to_time_strings(important_parts_from_text)}\n")

    important_parts = merged_interval_lists(important_parts_from_audio, important_parts_from_audio)

    #print(f"Audio Intervals: {intervals_to_time_strings(important_parts_from_audio)}\n")
    #print(f"Speech recognition Intervals: {intervals_to_time_strings(important_parts_from_text)}\n")

    #create_gif_from_intervals(video_file_path, important_parts)
    
    create_video_from_intervals(video_file_path, important_parts)


    #print(f"Audio Intervals: {intervals_to_time_strings(important_parts_from_audio)}\n")
    #print(f"Speech recognition Intervals: {intervals_to_time_strings(important_parts_from_text)}\n")
    #print(f"Merged Lists: {intervals_to_time_strings(important_parts)}\n")print(threshold)
    
    print(f"Merged Lists: {intervals_to_time_strings(important_parts)}\n")
    plot_rms_energy(file_name, rms_normalized, important_parts, 512, sample_rate,threshold,min_interval_duration, multiplier)
    print(f"Merged Lists: {intervals_to_time_strings(important_parts)}\n")

    os.remove(audio_file_path)


def create_video_from_intervals(video_path, intervals, video_duration=100):
    video = VideoFileClip(video_path)

    clips = []
    for start_time, end_time in intervals:
        clip = video.subclip(start_time, end_time)
        clip = clip.resize(width=320)  # Adjust the width to your desired resolution
        clips.append(clip)

    if len(clips) == 0:
        print("No clips found in the specified intervals. Exiting without creating a video.")
        return

    # Concatenate the clips
    concatenated_clips = concatenate_videoclips(clips)

    # Check if the total duration is less than the specified video_duration
    total_duration = concatenated_clips.duration
    if video_duration > total_duration:
        video_duration = total_duration

    # Create a short video from the concatenated clips
    output_video = concatenated_clips.subclip(0, video_duration)

    # Construct the output path for the video in the same folder as the input video
    video_output_path = os.path.splitext(video_path)[0] + "_output.mp4"

    # Save the video
    output_video.write_videofile(video_output_path, codec='libx264', fps=24)  # Adjust the fps as needed


def merged_interval_lists(list_from_audio, list_from_text):
    # Combine the two lists of intervals
    intervals = list_from_audio + list_from_text

    # Sort the list of intervals by start time
    intervals.sort(key=lambda x: x[0])

    # Create a new list to store the merged intervals
    merged_intervals = [intervals[0]]

    # Loop through each interval in the sorted list and merge overlapping intervals
    for interval in intervals[1:]:
        last_interval = merged_intervals[-1]
        if interval[0] <= last_interval[1]:
            merged_intervals[-1] = (last_interval[0], max(interval[1], last_interval[1]))
        else:
            merged_intervals.append(interval)

    return merged_intervals


def select_video_file():
    app = QApplication(sys.argv)
    video_file_types = "Video files (*.mp4 *.avi *.mkv *.flv *.mov *.wmv);;All files (*)"
    file_path, _ = QFileDialog.getOpenFileName(None, "Select a video file", "", video_file_types)
    app.quit()
    return file_path


def calc_rms_and_rms_normalized(audio):
    # Compute the root-mean-square (RMS) energy for each frame
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)

    # Normalize RMS energy
    rms_normalized = rms / np.max(rms)

    return rms, rms_normalized


def plot_rms_energy(file_name, rms_normalized, important_parts, hop_length, sample_rate, threshold, min_duration,threshold_multiplier):
    str_threshold = str(threshold)[0:5]
    str_min_duration = str(min_duration)[0:3]
    str_threshold_multiplier = str(threshold_multiplier)
    
    times = np.arange(rms_normalized.shape[1]) * hop_length / sample_rate
    plt.figure(figsize=(12, 4))
    plt.plot(times, rms_normalized[0], label="Normalized RMS Energy")

    for interval in important_parts:
        start_time, end_time = interval
        plt.axvspan(start_time, end_time, alpha=0.3, color='red', label='Important Interval')

    # Remove duplicate labels from the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel("Time (MM:SS)")
    plt.ylabel("Normalized RMS Energy")
    plt.title(f"{file_name} + \n Normalized RMS Energy Over Time with Important Intervals \n"
        + f"(threshold = {str_threshold}, min_duration = {str_min_duration}, str_threshold_multiplier ={str_threshold_multiplier})")

    # Format x-axis tick labels as MM:SS
    formatter = FuncFormatter(lambda seconds, _: seconds_to_mmss_string(seconds))
    plt.gca().xaxis.set_major_formatter(formatter)
    
    # Set more x-axis ticks
    plt.xticks(np.arange(0, max(times), step=max(times) / 10))

    # Display threshold and duration in the top left corner
    plt.text(times[0], 1, f"Threshold: {threshold}\nMin Duration: {min_duration}s", verticalalignment='top', fontsize=10)

    plt.show()


def find_important_parts(sample_rate, rms, rms_normalized, threshold, min_interval_duration=1.0):
    important_parts = []
    start_time = None
    for i, value in enumerate(rms_normalized[0]):
        if value > threshold and start_time is None:
            start_time = i * 512 / sample_rate
        elif value <= threshold and start_time is not None:
            end_time = i * 512 / sample_rate
            if end_time - start_time > min_interval_duration:
                important_parts.append((start_time, end_time))
            start_time = None

    updated_list = []
    for interval in important_parts:
        interval_duration = interval[1] - interval[0]
        if interval_duration < 14:
            new_interval = (interval[0]-(18 - interval_duration)/2, interval[1]+(10 - interval_duration)/2)
            updated_list.append(new_interval)
        else:
            updated_list.append(interval)

    updated_list = sorted(updated_list, key=lambda x: x[0])
    return updated_list


def calculate_mad_threshold(rms, multiplier=1.5):
    # Calculate the median of the RMS values
    median_rms = np.median(rms)

    # Calculate the Mean Absolute Deviation (MAD)
    mad = np.mean(np.abs(rms - median_rms))

    # Calculate the threshold using the median and MAD
    threshold = median_rms + multiplier * mad

    return threshold

def create_gif_from_intervals(video_path, intervals, gif_duration=100):
    video = VideoFileClip(video_path)

    clips = []
    for start_time, end_time in intervals:
        clip = video.subclip(start_time, end_time)
        clip = clip.resize(width=320)  # Adjust the width to your desired resolution
        clips.append(clip)

    if len(clips) == 0:
        print("No clips found in the specified intervals. Exiting without creating a GIF.")
        return

    # Concatenate the clips
    concatenated_clips = concatenate_videoclips(clips)

    # Check if the total duration is less than the specified gif_duration
    total_duration = concatenated_clips.duration
    if gif_duration > total_duration:
        gif_duration = total_duration

    # Create a short GIF from the concatenated clips
    gif = concatenated_clips.subclip(0, gif_duration)

    # Construct the output path for the GIF in the same folder as the video
    gif_output_path = os.path.splitext(video_path)[0] + ".gif"

    # Save the GIF
    gif.write_gif(gif_output_path, fps=30, program='imageio')  # Adjust the step value as needed


def seconds_to_mmss_string(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{int(seconds):02d}"


def getVideoAudio(video_file_path):
    # Extract the audio from the video file
    video = VideoFileClip(video_file_path)
    audio_file_path = os.path.splitext(video_file_path)[0] + ".wav"
    audio = video.audio
    audio.write_audiofile(audio_file_path)
    return audio, audio_file_path


def choose_video():
    # Create a GUI window for selecting the video file
    root = tk.Tk()
    root.withdraw()

    # Use absolute file paths to avoid issues with the working directory
    initial_dir = os.path.abspath(os.path.expanduser("~"))
    video_file_path = filedialog.askopenfilename(
        title="Select a video file", initialdir=initial_dir)

    # Check if the file exists before opening it
    if not os.path.exists(video_file_path):
        raise FileNotFoundError(f"File not found: {video_file_path}")

    # Extract the audio from the video file
    return video_file_path


def load_audio_file(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
    return audio, sample_rate


def important_words_list():
    words = ['Goal', 'Wow', 'Fantastic', 'Amazing', 'What a play', 'Incredible', 'Unbelievable', 'Sensational',
             'Magnificent', 'Spectacular', 'What a shot', 'Beautiful', 'Stunning', 'Game-changer!',
             "That's a classic!", 'This is unbelievable!', 'What a comeback!', 'Brilliant!',
             'This is why we love the game!', 'A moment of magic!', 'What a strike!', 'Superb!', 'Genius!',
             'World-class!', 'Pure quality!', 'What a finish!', 'Absolutely sensational!', "That's pure class!",
             'Breathtaking!', 'Top-notch!', 'Exquisite!', 'Simply amazing!', 'unbeatable', 'brilliant']

    lowercase_words = [word.lower() for word in words]
    no_exclamation_words = [word.replace('!', '') for word in lowercase_words]

    return lowercase_words


def transcribe_audio_file_to_console(audio_file, video_file_path, max_retries=1, retry_delay=3, chunk_length_ms=5000):

    video = VideoFileClip(video_file_path)
    important_words = important_words_list()
    r = sr.Recognizer()

    audio = AudioSegment.from_wav(audio_file)
    num_chunks = len(audio) // chunk_length_ms + int(len(audio) % chunk_length_ms > 0)
    transcribed_text = ""

    timestamps = []

    for i in range(num_chunks):
        start_time = i * chunk_length_ms
        end_time = (i + 1) * chunk_length_ms
        audio_chunk = audio[start_time:end_time]

        temp_chunk_file = f"temp_chunk_{i}.wav"
        audio_chunk.export(temp_chunk_file, format="wav")

        with sr.AudioFile(temp_chunk_file) as source:
            audio_data = r.record(source)

            retries = 0
            while retries < max_retries:
                try:
                    retries += 1
                    text = r.recognize_google(audio_data)
                    break
                except sr.UnknownValueError:
                    print(f"Google Speech Recognition could not understand audio chunk {i}/{num_chunks}")
                    if retries < max_retries - 1:
                        print(f"Retrying chunk {i}... (attempt {retries + 1})")
                        time.sleep(retry_delay)
                except sr.RequestError as e:
                    print(f"Error: {e}")
                    if retries < max_retries - 1:
                        print(f"Retrying chunk {i}... (attempt {retries + 1})")
                        time.sleep(retry_delay)
                else:
                    break
            else:
                print(f"Could not request results from Google Speech Recognition service for chunk {i} after multiple attempts.")
                os.remove(temp_chunk_file)
                continue

        transcribed_text += " \n" + text

        # Find the important words timestamp
        words = text.split()

        for j in range(len(words)):
            if words[j].lower() in important_words:
                timestamp = i * (chunk_length_ms/1000) + (j * (len(words)/(chunk_length_ms/1000)))
                timestamps.append(int(timestamp))

        os.remove(temp_chunk_file)

    # Create intervals around each timestamp
    intervals = [(int(time - 2), int(time + 2)) for time in timestamps]

    # write the text to a file with the same name as the audio file, but with a ".txt" extension
    text_file = os.path.splitext(audio_file)[0] + ".txt"
    with open(text_file, "w") as f:
        f.write(transcribed_text)

    return intervals

def intervals_to_time_strings(intervals):
    """Convert a list of time intervals in seconds to a list of time strings in MM:SS format."""
    time_strings = []
    for interval in intervals:
        start_time = int(interval[0])
        end_time = int(interval[1])
        start_time_str = f"{start_time // 60:02d}:{start_time % 60:02d}"
        end_time_str = f"{end_time // 60:02d}:{end_time % 60:02d}"
        time_strings.append(f"{start_time_str}-{end_time_str}")
    return time_strings



if __name__ == "__main__":
    main()
