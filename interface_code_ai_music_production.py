import os
import torch
import gradio as gr
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import subprocess as sp
import sys
from typing import List, Optional
from pathlib import Path
import io
import select
import tempfile
import shutil
import logging
import time
import csv
from datetime import datetime


# Global variables
current_language = "english"
generation_count = 0 # Generation counters
generated_files = []
history = []
SESSIONS_BASE_DIR = None  # Main directory for all sessions
CURRENT_SESSION_DIR = None  # Directory for current session
total_tracks_generated = 0  # Tracks generated in the session
total_tracks_separated = 0  # Tracks user actually separated

#N.B. it is possible to add support for further languages
# Text for the menu 
texts = {
    "english": {
        "title": "Instrumental Music Generator with MusicGEN and Track Separation",
        "description_label": "Enter a description of the track",
        "description_placeholder": "An indie pop song, with catchy melodies and hard-hitting drums",
        "model_label": "Select MusicGEN Model",
        "duration_label": "Duration (seconds)",
        "generate_button": "Generate Music",
        "separation_title": "Track Separation",
        "change_language": " ",
        "separate_button": "Separate Audio Clip",
        "separate_all_button": "Separate All",
        "counter_label":"N# Generations: ",
    },
}



# Change language function
def change_language():
    global current_language
    current_language = "english"
    updated_texts = texts[current_language]

    # Update texts for separation buttons
    separate_buttons_updates = [
        gr.update(value=f"{updated_texts['separate_button']} {i+1}") for i in range(3)
    ]

    return (
        gr.update(value=updated_texts["title"]),
        gr.update(value=updated_texts["change_language"]),
        gr.update(
            label=updated_texts["description_label"],
            placeholder=updated_texts["description_placeholder"]
        ),
        gr.update(label=updated_texts["model_label"]),
        gr.update(label=updated_texts["duration_label"]),
        gr.update(value=updated_texts["generate_button"]),
        gr.update(value=updated_texts["separation_title"]),
        gr.update(value=f"{updated_texts['counter_label']} {generation_count}"),
        *separate_buttons_updates,
        gr.update(value="Separate All"),  
    )


# Updates texts depending on language
def update_texts():
    return texts[current_language]

# Computational device configuration (CUDA or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Main directory where data are stored
BASE_DIR = ""

def setup_logging():
    global CURRENT_SESSION_DIR

    # Ensure the main Sessions/ directory exists
    SESSIONS_BASE_DIR = os.path.join(BASE_DIR, "Sessions")
    os.makedirs(SESSIONS_BASE_DIR, exist_ok=True)

    # Create the session folder INSIDE Sessions/
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    CURRENT_SESSION_DIR = os.path.join(SESSIONS_BASE_DIR, f"session_{now}")
    os.makedirs(CURRENT_SESSION_DIR, exist_ok=True)

    # Set up logging inside the session folder
    log_file = os.path.join(CURRENT_SESSION_DIR, "session.log")


    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=== New session started ===")

    # Create per-session CSV
    session_csv = os.path.join(CURRENT_SESSION_DIR, "session.csv")
    global_csv = os.path.join(SESSIONS_BASE_DIR, "all_sessions.csv")  # Global CSV

    
    for csv_file in [session_csv, global_csv]:
        if not os.path.exists(csv_file):  # Only write headers if file doesn't exist
            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Generation ID", "Timestamp", "Description", "Duration (s)", "Model", 
                             "Processing Time (s)", "Generated Files"])

    logging.info(f"CSV log files created: {session_csv}, {global_csv}")


# Function to load musicgen model
def load_model(model_name: str) -> MusicGen:
    full_model_name = f"facebook/musicgen-{model_name}"
    print(f"Loading model {full_model_name}...")
    return MusicGen.get_pretrained(full_model_name)


# Function to perform music generation
def generate_music(description: str, duration: int, model_name: str) -> tuple[str, str, str, str]:
    global generation_count, history, SESSIONS_BASE_DIR, CURRENT_SESSION_DIR, total_tracks_generated

    if CURRENT_SESSION_DIR is None:
        setup_logging()

    # Ensure timestamp is defined at the start
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    # Ensure CSV paths are defined
    session_csv = os.path.join(CURRENT_SESSION_DIR, "session.csv")
    global_csv = os.path.join(BASE_DIR, "Sessions", "all_sessions.csv")

    # Increment generation count
    generation_count += 1
    logging.info(f"Generation #{generation_count} started. Description: '{description}', Duration: {duration}s, Model: '{model_name}'")
    
    start_time = time.time()  # Start measuring time

    # Create folder for this generation
    generation_dir = os.path.join(CURRENT_SESSION_DIR, f"generation_{generation_count}")
    os.makedirs(generation_dir, exist_ok=True)

    try:
        # Load the model safely
        model = load_model(model_name)
        model.set_generation_params(duration=duration)

        descriptions = [description] * 3
        wavs = model.generate(descriptions)

        # Ensure `wavs` is valid
        if isinstance(wavs, torch.Tensor) and wavs.numel() == 0:
            raise ValueError("Model output is empty")


        file_paths = []
        for idx, wav in enumerate(wavs):
            file_path = audio_write(
                os.path.join(generation_dir, f"output_{idx}_{os.path.basename(CURRENT_SESSION_DIR)}"),
                wav.cpu(),
                model.sample_rate,
                strategy="loudness"
            )
            file_paths.append(file_path)
            print(f" Saved file: {file_path}")  # 🔍 Debugging print

        for file in file_paths:
            if os.path.isdir(file):
                print(f" ERROR: {file} is a directory, not a file!")
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)

        logging.info(f"Generation #{generation_count} completed in {elapsed_time} seconds. Generated files: {file_paths}")

        # Write to CSV safely
        try:
            with open(session_csv, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([generation_count, timestamp, description, duration, model_name, elapsed_time, str(file_paths)])
        except Exception as e:
            print(f" Error writing to CSV {session_csv}: {e}")

        try:
            with open(global_csv, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([generation_count, timestamp, description, duration, model_name, elapsed_time, str(file_paths)])
        except Exception as e:
            print(f" Error writing to CSV {global_csv}: {e}")

        # Update history
        history.append({
            "id": generation_count,
            "files": file_paths,
            "description": description,
        })

        # Track the number of generated tracks
        total_tracks_generated += len(file_paths)

        # Avoid index errors when returning
        while len(file_paths) < 3:
            file_paths.append("")  # Fill missing slots with empty strings

        print(f" Generated files: {file_paths}")
        if os.path.isdir(file_paths[0]):
                print(f" ERROR: {file_paths[0]} is a directory!")


        return (
            file_paths[0] if os.path.exists(file_paths[0]) else "",  
            file_paths[1] if os.path.exists(file_paths[1]) else "",  
            file_paths[2] if os.path.exists(file_paths[2]) else "",  
            f"{texts[current_language]['counter_label']} {generation_count}"
        )

    except Exception as e:
        logging.error(f"Generation failed: {e}")
        print(f" Error: {e}")
        return "", "", "", f"Error: {e}"

def create_session_folder():
    global CURRENT_SESSION_DIR
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    CURRENT_SESSION_DIR = os.path.join(BASE_DIR, f"session_{now}")
    os.makedirs(CURRENT_SESSION_DIR, exist_ok=True)
    print(f"Session folder created: {CURRENT_SESSION_DIR}")

# History update function
def history_update() -> List[List[str]]:
    if not history:
        return [["-1", "No generated item available"]]
    return [[str(item["id"]), item["description"]] for item in history]

def separate_all_wrapper(clip1, clip2, clip3):
    """
    Wrapper to call separate_all_clips with the current value of generation_count.
    """
    return separate_all_clips(clip1, clip2, clip3, generation_count)

# File loading function from history
def load_from_history(id_generation: int) -> tuple[Optional[str], Optional[str], Optional[str]]:
    for item in history:
        if item["id"] == id_generation:
            return tuple(item["files"])  # Returns files from selected generation
    return None, None, None  # No file found

# Utility function to copy process streams
def copy_process_streams(process: sp.Popen):
    def raw(stream: Optional[io.BufferedIOBase]) -> io.BufferedIOBase:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd = {
        p_stdout.fileno(): (p_stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())

    while fds:
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            p_stream, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2 ** 16)
            if not raw_buf:
                fds.remove(fd)
                continue
            buf = raw_buf.decode()
            std.write(buf)
            std.flush()

# Separate tracks with demucs
def separate_tracks(file_audio_path: str, generation_id: int, clip_index: int) -> List[Optional[str]]:
    
    if not os.path.exists(file_audio_path):
        print("Error: file not found")
        return [None] * 5

    # subfolder to save generation output
    generation_dir = os.path.join(CURRENT_SESSION_DIR, f"generation_{generation_id}")
    separation_dir = os.path.join(generation_dir, f"STEMS")
    os.makedirs(separation_dir, exist_ok=True)

    # Runs demucs to separate tracks
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-o", separation_dir,  # Demucs-generated output
        "-n", "htdemucs_6s",   #  Demucs model version
        file_audio_path    # File to which separation is applied
    ]
    print(f"Apply separation to file: {file_audio_path}")
    p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    copy_process_streams(p)
    p.wait()

    if p.returncode != 0:
        print("Error during separation.")
        return [None] * 5

    # Output path directory generated by DEMUCS
    demucs_output_dir = os.path.join(separation_dir, "htdemucs_6s", os.path.splitext(os.path.basename(file_audio_path))[0])
    
    # paths of the last files generated by demucs
    stems = ["drums", "bass", "guitar", "piano", "other"]
    final_paths = [os.path.join(demucs_output_dir, f"{stem}.wav") for stem in stems]

    # Checks that files ( None for missing files)
    return [path if os.path.exists(path) else None for path in final_paths]

def separate_and_path_check(file_audio_path: Optional[str], generation_id: int, clip_index: int) -> tuple[Optional[str], ...]:
    """
    Check and separate an audio file. Returns the paths of the separate tracks.
    """
    if not file_audio_path or not os.path.exists(file_audio_path):
        errore = "Error: No valid tracks found for separation. Generate music first!"
        return None, None, None, None, None, errore

    output_files = separate_tracks(file_audio_path, generation_id, clip_index)
    return *output_files, ""

def separate_all_clips(audio_clip_1: Optional[str], audio_clip_2: Optional[str], audio_clip_3: Optional[str], generation_id: int) -> tuple:
    """
    Automatic separation of all generated audio clips.
    """
    global total_tracks_separated, total_tracks_generated


    separation_results = []
    for idx, clip in enumerate([audio_clip_1, audio_clip_2, audio_clip_3]):
        if clip:
            separation = separate_tracks(clip, generation_id, idx)
            separation = [str(path) if path else None for path in separation]  # Ensure paths are strings
            separation_results.extend(separation)
            total_tracks_separated += 1

        else:
            separation_results.extend([None, None, None, None, None])  

    # Prevent division by zero and round percentage
    separation_usage_rate = round((total_tracks_separated / total_tracks_generated * 100), 2) if total_tracks_generated > 0 else 0

    logging.info(f"Session Summary: {total_tracks_generated} tracks generated, {total_tracks_separated} separated, {separation_usage_rate:.2f}% usage rate.")

    # Define paths for CSV files
    session_csv = os.path.join(CURRENT_SESSION_DIR, "session.csv")
    global_csv = os.path.join(BASE_DIR, "Sessions", "all_sessions.csv")

    # Write to session.csv safely
    try:
        file_exists = os.path.exists(session_csv) and os.stat(session_csv).st_size > 0
        with open(session_csv, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:  # If file is new, write headers
                writer.writerow(["Session Summary", "Tracks Generated", "Tracks Separated", "Separation Usage Rate (%)"])
            writer.writerow(["Session Summary", total_tracks_generated, total_tracks_separated, f"{separation_usage_rate:.2f}%"])
        print(f" Session summary saved in {session_csv}")  # Debugging print
    except Exception as e:
        print(f" Error writing session summary to {session_csv}: {e}")

    # Read previous session statistics
    previous_sessions = []
    try:
        if os.path.exists(global_csv) and os.stat(global_csv).st_size > 0:
            with open(global_csv, mode="r", newline="") as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 4:
                        previous_sessions.append(float(row[3].replace('%', '')))
    except Exception as e:
        print(f" Warning: Could not read {global_csv}: {e}")

    # Compute overall separation rate correctly
    all_sessions_separation_rate = round(
        (sum(previous_sessions) + separation_usage_rate) / (len(previous_sessions) + 1),
        2
    ) if previous_sessions else separation_usage_rate

    # Write to global CSV safely
    try:
        file_exists = os.path.exists(global_csv) and os.stat(global_csv).st_size > 0
        with open(global_csv, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:  # If empty, write headers
                writer.writerow(["Session Summary", "Tracks Generated", "Tracks Separated", "Overall Separation Usage Rate (%)"])
            writer.writerow(["Session Summary", total_tracks_generated, total_tracks_separated, f"{all_sessions_separation_rate:.2f}%"])
        print(f" Updated all_sessions.csv with new average separation rate.")
    except Exception as e:
        print(f" Error updating all_sessions.csv: {e}")

    return tuple(separation_results)



# Graphical interface
with gr.Blocks() as demo:

    # Main area with Tabs
    with gr.Tabs():
        
        # Tab for music generation
        with gr.Tab("Generation") as tab_generation:
            title = gr.Markdown(texts[current_language]["title"])
            counter_label = gr.Markdown(f"{texts[current_language]['counter_label']} {generation_count}")

            with gr.Row():
                description = gr.Textbox(
                    lines=2,
                    placeholder=texts[current_language]["description_placeholder"],
                    label=texts[current_language]["description_label"],
                )
            with gr.Row():
                model_choice = gr.Radio(
                    choices=["small", "medium", "large"],
                    value="small",
                    label=texts[current_language]["model_label"],
                )
            with gr.Row():
                duration = gr.Slider(
                    minimum=1, maximum=30, value=10, step=1,
                    label=texts[current_language]["duration_label"]
                )
            with gr.Row():
                
                generate_button = gr.Button(
                    texts[current_language]["generate_button"],
                    elem_id="generate_button",
                    variant="primary"
                )
            with gr.Row():
                output_audio = [gr.Audio(type='filepath', label=f"Clip Audio {i+1}",show_download_button=True) for i in range(3)]
            
            # Linking the Button to the Textbox
            description.submit(
                fn=generate_music,  # Funzione di generazione
                inputs=[description, duration, model_choice],  # function input
                outputs=output_audio + [counter_label],  # function output
            )

            # Linking the Button to music generation
            generate_button.click(
                fn=generate_music,
                inputs=[description, duration, model_choice],
                outputs=output_audio + [counter_label],  # Three audios + counter
            )

        # Tab for tracks separation
        with gr.Tab("Separation") as tab_separation:
            separation_title = gr.Markdown(texts[current_language]["separation_title"])

            separate_buttons = []  # list for separation-related buttons
            stems_outputs = []

            separate_all_button = gr.Button(texts[current_language].get("separate_all_button", "Separate All"), variant="primary")

            for i in range(3):  # For each generated audio clip
                with gr.Group():
                    with gr.Row():
                        separate_button = gr.Button(
                            f"{texts[current_language]['separate_button']} {i+1}"
                        )
                        separate_buttons.append(separate_button)  # Save buttons

                    with gr.Row():
                        # Output for separated stems (Drums, Bass, ecc.)
                        stem_audios = [
                            gr.Audio(label=stem.capitalize(), type='filepath')
                            for stem in ["Drums", "Bass", "Guitar", "Piano", "Other"]
                        ]
                        stems_outputs.append(stem_audios)
                    separation_output = gr.Markdown("")  # Error messages

                    # Connect single separation buttons
                    separate_button.click(
                        fn=lambda file_audio_path=output_audio[i]: separate_and_path_check(file_audio_path,generation_count, i),
                        inputs=[output_audio[i]],
                        outputs=stem_audios + [separation_output],  # Output audio + messages
                    )


                    # Link button "Separate All"
                    separate_all_button.click(
                        fn=lambda clip1, clip2, clip3: separate_all_clips(clip1, clip2, clip3, generation_count),
                        inputs=[output_audio[0], output_audio[1], output_audio[2]],
                        outputs=[stem_audio for audio_group in stems_outputs for stem_audio in audio_group]
                    )

        # Tab for history
        
        with gr.Tab("History", visible=False) as tab_history:
            gr.Markdown("### Generation history")

            # Dataset to show list of generations
            history_list = gr.Dataset(components=[
                gr.Textbox(label="ID Generation"),
                gr.Textbox(label="Description")
            ])

            # Area to show audio files selected from history
            with gr.Row():
                history_audio = [gr.Audio(type='filepath', label=f"Clip Audio {i+1}") for i in range(3)]

            # Button to update history
            update_button = gr.Button("History Update")

            # Button to update history
            update_button.click(
                fn=history_update,
                inputs=[],
                outputs=[history_list],  # Generation list
            )

            # Load a new generation
            load_id = gr.Number(label="Generation ID", precision=0)
            load_button = gr.Button("Load generation")

            load_button.click(
                fn=load_from_history,
                inputs=[load_id],
                outputs=history_audio,  # Show audio files of the selected generation
            )

    # Spacing and button to change language
    with gr.Row():
        gr.Markdown(" ")
    with gr.Row():
        change_language_button = gr.Button(
            texts[current_language]["change_language"],
            elem_id="language_button",
            variant="secondary"
        )

    # Connect language change
    change_language_button.click(
        fn=change_language,
        inputs=[],
        outputs=[
            title, # Updated title
            change_language_button, # updated button
            description, # Updated textbox
            model_choice, # Updated radio
            duration, # Updated slider
            generate_button, # Updated generate music button
            separation_title, # Updated track separation title
            *separate_buttons, # Updated separation buttons
        ],
    )

# Run the interface
print("Before demo.launch()")
demo.launch(share=False, server_port=7860)
print("After demo.launch()")