import streamlit as st
import numpy as np
import io
from preprocess import SEQUENCE_LENGTH
from melodygenerator import MelodyGenerator  # Import your MelodyGenerator class
import pretty_midi
from pydub import AudioSegment


NOTE_MAPPINGS = {
    "C4": 60, "C#4": 61, "D4": 62, "D#4": 63, "E4": 64, "F4": 65,
    "F#4": 66, "G4": 67, "G#4": 68, "A4": 69, "A#4": 70, "B4": 71,
    "C5": 72, "D5": 74, "E5": 76, "F5": 77, "G5": 79, "A5": 81,"HOLD":"_","REST":"r"
}
NOTE_OPTIONS=[f"{note} (MIDI {midi})" for note, midi in NOTE_MAPPINGS.items()]



# Sine wave synthesis for each MIDI note
def synthesize_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    return wave

# Convert MIDI file to a synthesized WAV-like byte stream
def midi_to_wav(midi_data):
    midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_data))
    audio_data = np.array([])

    for instrument in midi.instruments:
        for note in instrument.notes:
            freq = pretty_midi.note_number_to_hz(note.pitch)
            duration = note.end - note.start
            wave = synthesize_wave(freq, duration)
            audio_data = np.concatenate([audio_data, wave])

    # Convert numpy array to AudioSegment
    audio = AudioSegment(
        data=(audio_data * 32767).astype(np.int16).tobytes(),
        sample_width=2,
        frame_rate=44100,
        channels=1
    )
    
    # Export to byte stream
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    return wav_io

# Streamlit app
def main():
    st.title("Melody Generator and Player")

    #-------------------------------------------------------------------------------------
    seed=""

    if 'selections' not in st.session_state:
        st.session_state.selections = []


    with st.form(key='note_form'):
        selected_note = st.selectbox("Select a note", options=NOTE_OPTIONS)
        submit_button = st.form_submit_button(label='Add Note')

        if submit_button:
            st.session_state.selections.append(selected_note)

    st.write("Selected Notes and Durations:")
    for note in st.session_state.selections:
        note=note.split()[0]
        seed+=str(NOTE_MAPPINGS[note])
        st.write(seed)
        seed+=" "


    # seed = st.text_input("Enter seed melody:", "67 _ 67 _ 65 64 _ _")
    temperature = st.slider("Select temperature:", 0.0, 1.0, 0.5)
    
    if st.button("Generate and Play Melody"):
        mg = MelodyGenerator()
        melody = mg.generate_melody(seed, num_steps=500, max_sequence_length=SEQUENCE_LENGTH, temperature=temperature)
        
        # Save melody as a MIDI file
        midi_file = mg.save_melody(melody)

        # Convert MIDI to WAV
        wav_file = midi_to_wav(midi_file.read())

        # Play the audio
        st.audio(wav_file, format="audio/wav")

        # Download option
        st.download_button("Download WAV", wav_file, file_name="generated_melody.wav", mime="audio/wav")

if __name__ == "__main__":
    main()
