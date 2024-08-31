import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def residual_block(input_tensor, filters=256, kernel_size=3):
    # First convolutional layer
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Second convolutional layer
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Add skip connection
    x = tf.keras.layers.Add()([x, input_tensor])
    x = tf.keras.layers.ReLU()(x)
    
    return x

class Connect4NN:
    def __init__(self, row_count, col_count):
        # Define the input layer
        input_layer = tf.keras.layers.Input(shape=(row_count, col_count, 1))
        
        # Initial convolutional layer
        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same')(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        # Add residual blocks
        for _ in range(20):
            x = residual_block(x, filters=256)

        # Policy head
        policy_head = tf.keras.layers.Conv2D(2, kernel_size=1)(x)
        policy_head = tf.keras.layers.BatchNormalization()(policy_head)
        policy_head = tf.keras.layers.ReLU()(policy_head)
        policy_head = tf.keras.layers.Flatten()(policy_head)  # Flatten before Dense
        policy_head = tf.keras.layers.Dense(7, activation='softmax', name='policy_output')(policy_head)

        # Value head
        value_head = tf.keras.layers.Conv2D(1, kernel_size=1)(x)
        value_head = tf.keras.layers.BatchNormalization()(value_head)
        value_head = tf.keras.layers.ReLU()(value_head)
        value_head = tf.keras.layers.Flatten()(value_head)  # Flatten before Dense
        value_head = tf.keras.layers.Dense(1, activation='tanh', name='value_output')(value_head)

        # Define the model
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=[policy_head, value_head])

        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss={'policy_output': 'categorical_crossentropy', 'value_output': 'mean_squared_error'},
            metrics={'policy_output': 'accuracy', 'value_output': 'mse'}
        )