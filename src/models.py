from keras.models import Model
from keras.layers import Input, Dropout, Merge, TimeDistributed, Masking, Dense
from keras.layers.recurrent import LSTM
from keras.regularizers import l2

def NIC(max_token_length, vocabulary_size, num_image_features=2048,
        hidden_size=512, embedding_size=512, regularizer=1e-8):

    # word embedding
    text_input = Input(shape=(max_token_length, vocabulary_size), name='text')
    text_mask = Masking(mask_value=0.0, name='text_mask')(text_input)
    text_to_embedding = TimeDistributed(Dense(output_dim=embedding_size,
                                        input_dim = vocabulary_size,
                                        W_regularizer = l2(regularizer),
                                        name = 'text_embedding'))(text_mask)

    text_dropout = Dropout(.5, name='text_dropout')(text_to_embedding)

    # image embedding
    image_input = Input(shape=(max_token_length, num_image_features),
                                                        name='image')
    image_embedding = TimeDistributed(Dense(output_dim=embedding_size,
                                        input_dim=num_image_features,
                                        W_regularizer=l2(regularizer),
                                        name='image_embedding'))(image_input)
    image_dropout = Dropout(.5,name='image_dropout')(image_embedding)

    # language model
    recurrent_inputs = [text_dropout, image_dropout]
    merged_input = Merge(mode='sum')(recurrent_inputs)
    recurrent_network = LSTM(output_dim=hidden_size,
                            input_dim=embedding_size,
                            W_regularizer=l2(regularizer),
                            U_regularizer=l2(regularizer),
                            return_sequences=True,
                            name='recurrent_network')(merged_input)

    output = TimeDistributed(Dense(output_dim=vocabulary_size,
                                    input_dim=hidden_size,
                                    W_regularizer=l2(regularizer),
                                    activation='softmax'),
                                    name='output')(recurrent_network)

    inputs = [text_input, image_input]
    model = Model(input=inputs,output=output)
    return model

