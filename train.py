import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH


VOCABULARY_SIZE = 38  # Adjust based on your actual vocabulary size (e.g., MIDI note + rest + "_")
EMBEDDING_DIM = 16
NUM_UNITS = [256, 125] #in layer1, layer2
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 90
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"


def build_model(vocabulary_size, embedding_dim, num_units, loss, learning_rate):
    """Builds and compiles model

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply

    :return model (tf model)
    """

    # create the model architecture
    input = keras.layers.Input(shape=(SEQUENCE_LENGTH))
    x= keras.layers.Embedding(input_dim= vocabulary_size, output_dim=embedding_dim)(input)
    x = keras.layers.LSTM(num_units[0],return_sequences=True)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LSTM(num_units[1])(x)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(units= vocabulary_size, activation="softmax")(x)

    model = keras.Model(input, output)

    # compile model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    model.summary()

    return model


def train(output_units=VOCABULARY_SIZE, embedding_dim= EMBEDDING_DIM,num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    """Train and save TF model.

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply
    """

    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # build the network
    model = build_model(output_units, embedding_dim, num_units, loss, learning_rate)

    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()