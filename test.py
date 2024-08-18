import streamlit as st
import pretty_midi
import numpy as np
from pydub import AudioSegment
import io

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
st.title("MIDI to Audio Converter")

uploaded_midi = st.file_uploader("Upload MIDI file", type=["mid", "midi"])

if uploaded_midi is not None:
    midi_bytes = uploaded_midi.read()

    # Convert MIDI to WAV
    wav_file = midi_to_wav(midi_bytes)

    # Play the audio
    st.audio(wav_file, format="audio/wav")

    # Download option
    st.download_button("Download WAV", wav_file, file_name="output.wav", mime="audio/wav")
