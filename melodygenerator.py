import json
import numpy as np
import tensorflow as tf  
from tensorflow import keras  
import music21 as m21
import io
import tempfile
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies."""

    def __init__(self, model_path="model.h5"):
        """Constructor that initialises TensorFlow model"""

        self.model_path = model_path
        self.model = keras.models.load_model(model_path,compile=False)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """Generates a melody using the DL model and returns a midi file.

        :param seed (list): Melody seed with the notation used to encode the dataset
        :param num_steps (int): Number of steps to be generated
        :param max_sequence_len (int): Max number of steps in seed to be considered for generation
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return melody (list of str): List with symbols representing a melody
        """

        # create seed with start symbols
        seed=seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]
 
        for _ in range(num_steps):

            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # update melody
            melody.append(output_symbol)

        return melody


    def _sample_with_temperature(self, probabilites, temperature):
        """Samples an index from a probability array reapplying softmax using temperature

        :param predictions (nd.array): Array containing probabilities for each of the possible outputs.
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return index (int): Selected output symbol
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index


    def save_melody(self, melody, step_duration=0.25):
        """Converts a melody into a MIDI file

        :param melody (list of str):
        :param min_duration (float): Duration of each time step in quarter length
        :param file_name (str): Name of midi file
        :return:
        """

        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        #stream.write(format, file_name)

        # midi_file = io.BytesIO()
        # stream.write('midi', fp=midi_file)
        # midi_file.seek(0)

        # return midi_file

         # Write the stream to a temporary file on disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as temp_midi_file:
            stream.write('midi', fp=temp_midi_file.name)

        # Read the temporary file into a BytesIO object
        with open(temp_midi_file.name, 'rb') as f:
            midi_file = io.BytesIO(f.read())

        # # Clean up the temporary file
        # os.remove(temp_midi_file.name)

        # midi_file.seek(0)  # Reset the file pointer to the start
        return midi_file


# if __name__ == "__main__":
#     mg = MelodyGenerator()
#     seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
#     seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
#     melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.3)
#     print(melody)
#     mg.save_melody(melody)

def main():
    # Example usage
    seed = "67 _ 67 _ 65 64 _ _"
  
    mg = MelodyGenerator()
    melody = mg.generate_melody(seed, num_steps=500, max_sequence_length=SEQUENCE_LENGTH, temperature=0.5)
    
    # Save melody as a MIDI file
    midi_file = mg.save_melody(melody)
    
    # Save to disk for standalone use
    with open("generated_melody.mid", "wb") as f:
        f.write(midi_file.read())
    
    print("Generated melody saved to 'generated_melody.mid'.")


if __name__ == "__main__":
    main()



