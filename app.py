import streamlit as st
import numpy as np
import io
from preprocess import SEQUENCE_LENGTH
from melodygenerator import MelodyGenerator  
import pretty_midi
from pydub import AudioSegment
from scipy.signal import butter, filtfilt


NOTE_MAPPINGS = {
    "HOLD":"_","REST":"r","C4": 60, "C#4": 61, "D4": 62, "D#4": 63, "E4": 64, "F4": 65,
    "F#4": 66, "G4": 67, "G#4": 68, "A4": 69, "A#4": 70, "B4": 71,
    "C5": 72, "D5": 74, "E5": 76, "F5": 77, "G5": 79, "A5": 81
}
NOTE_OPTIONS=[f"{item}" for item in NOTE_MAPPINGS.keys()]



# Advanced sine wave synthesis with ADSR envelope to smooth transitions
def synthesize_wave(frequency, duration, sample_rate=44100, fade_duration=0.05, attack=0.01, release=0.05):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Apply ADSR envelope
    attack_samples = int(attack * sample_rate)
    release_samples = int(release * sample_rate)
    sustain_samples = len(wave) - (attack_samples + release_samples)

    envelope = np.concatenate([
        np.linspace(0, 1, attack_samples) ** 2,  # Attack phase with smoother ramp-up
        np.ones(sustain_samples),               # Sustain phase (constant)
        np.linspace(1, 0, release_samples) ** 2 # Release phase with smoother ramp-down
    ])

    wave *= envelope

    return wave

# Apply a high-pass filter to remove low-frequency noise (e.g., subsonic clicks)
def apply_highpass_filter(audio_data, sample_rate, cutoff=100, order=3):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_wave = filtfilt(b, a, audio_data)
    return filtered_wave

# Apply a low-pass filter to smooth the waveform
def apply_filter(audio_data, sample_rate, cutoff=5000, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_wave = filtfilt(b, a, audio_data)
    return filtered_wave

# Convert MIDI file to a synthesized WAV-like byte stream with enhanced filtering and volume boosting
def midi_to_wav(midi_data, sample_rate=44100, highpass_cutoff=100, filter_cutoff=5000, boost_db=5):
    midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_data))
    audio_data = np.array([])

    for instrument in midi.instruments:
        for note in instrument.notes:
            freq = pretty_midi.note_number_to_hz(note.pitch)
            duration = note.end - note.start
            
            # Generate the wave for the note with ADSR envelope
            wave = synthesize_wave(freq, duration, sample_rate=sample_rate)
            
            # Concatenate the new wave to the overall audio
            audio_data = np.concatenate([audio_data, wave])

    # Apply high-pass filter to remove low-frequency transients
    audio_data = apply_highpass_filter(audio_data, sample_rate, cutoff=highpass_cutoff)

    # Apply low-pass filter to smooth the waveform
    audio_data = apply_filter(audio_data, sample_rate, cutoff=filter_cutoff)

    # Normalize the audio to prevent clipping
    audio_data /= np.max(np.abs(audio_data))
    
    # Pre-zero padding to reduce initial transient noise
    silence_padding = np.zeros(int(0.05 * sample_rate))  # 50 ms silence
    audio_data = np.concatenate([silence_padding, audio_data])

    # Convert numpy array to AudioSegment and apply volume boost
    audio = AudioSegment(
        data=(audio_data * 32767).astype(np.int16).tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1
    ).apply_gain(boost_db)
    
    # Export to byte stream
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    return wav_io

# Streamlit app
def main():
    st.title("Melody Generator")

    # Initialize session state for selected notes
    if 'selections' not in st.session_state:
        st.session_state.selections = []

    # Form to input notes for the melody seed
    with st.form(key='note_form'):
        selected_note = st.selectbox("Select a note", options=NOTE_OPTIONS)
        submit_button = st.form_submit_button(label='Add Note')

        if submit_button:
            st.session_state.selections.append(selected_note)

    # Display selected notes and generate the seed string
    st.write("Selected Notes and Durations:")
    seed = ""
    for note in st.session_state.selections:
        note = note.split()[0]
        seed += str(NOTE_MAPPINGS[note])
        st.write(note)
        seed += " "

    # Creativity temperature slider
    temperature = st.slider("Set Creativity Value (where 0 = less creative, 1 = more creative):", 0.0, 1.0, 0.5)
    
    if st.button("Generate"):
        mg = MelodyGenerator()
        melody = mg.generate_melody(seed, num_steps=500, 
        max_sequence_length=SEQUENCE_LENGTH, temperature=temperature)
        
        # Save the melody as a MIDI file
        midi_file = mg.save_melody(melody)

        # Convert the MIDI file to WAV
        wav_file = midi_to_wav(midi_file.read())

        # Play the generated WAV file
        st.audio(wav_file, format="audio/wav")

        # Provide options to download both the MIDI and WAV files
        st.download_button("Download WAV", wav_file, file_name="generated_melody.wav", mime="audio/wav")

        # Rewind the MIDI file for download after reading it
        midi_file.seek(0)
        st.download_button("Download MIDI", midi_file, file_name="generated_melody.mid", mime="audio/midi")


if __name__ == "__main__":
    main()
