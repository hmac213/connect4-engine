import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def squeeze_excite_block(input_tensor, ratio=16):
    """Squeeze and Excitation block for channel attention"""
    filters = input_tensor.shape[-1]
    se_shape = (1, 1, filters)
    
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Reshape(se_shape)(se)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    
    return tf.keras.layers.multiply([input_tensor, se])

def residual_block(input_tensor, filters=128, kernel_size=3):
    """Residual block with proper projection shortcut"""
    # Projection shortcut if dimensions don't match
    if input_tensor.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')(input_tensor)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    # First convolutional layer
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Second convolutional layer
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Add skip connection
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    
    # Add squeeze and excitation with reduced ratio
    x = squeeze_excite_block(x, ratio=8)  # Reduced ratio for smaller networks
    
    return x

class Connect4NN:
    def __init__(self, row_count, col_count):
        # Define the input layer - 3 channels
        input_layer = tf.keras.layers.Input(shape=(row_count, col_count, 3))
        
        # Initial convolutional layer
        x = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same')(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        # Add residual blocks with squeeze and excitation
        for _ in range(8):  # 8 blocks is sufficient for Connect4
            x = residual_block(x, filters=128)
            x = tf.keras.layers.Dropout(0.1)(x)

        # Simplified policy head
        policy_head = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same')(x)
        policy_head = tf.keras.layers.BatchNormalization()(policy_head)
        policy_head = tf.keras.layers.ReLU()(policy_head)
        policy_head = tf.keras.layers.Conv2D(1, kernel_size=1)(policy_head)
        policy_head = tf.keras.layers.Flatten()(policy_head)
        policy_head = tf.keras.layers.Dense(7, activation='softmax', name='policy_output')(policy_head)

        # Simplified value head
        value_head = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same')(x)
        value_head = tf.keras.layers.BatchNormalization()(value_head)
        value_head = tf.keras.layers.ReLU()(value_head)
        value_head = tf.keras.layers.GlobalAveragePooling2D()(value_head)
        value_head = tf.keras.layers.Dense(64, activation='relu')(value_head)
        value_head = tf.keras.layers.Dropout(0.2)(value_head)
        value_head = tf.keras.layers.Dense(1, activation='tanh', name='value_output')(value_head)

        # Define the model
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=[policy_head, value_head])

        # Compile with lower initial learning rate
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'policy_output': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                'value_output': 'mean_squared_error'
            },
            loss_weights={
                'policy_output': 1.0,
                'value_output': 1.0
            },
            metrics={
                'policy_output': 'accuracy',
                'value_output': 'mse'
            }
        )