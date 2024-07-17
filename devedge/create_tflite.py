import tensorflow as tf
from keras import backend as K
import argparse
import os

def load_model(model_path):
    def r_loss(y_true, y_pred):
        return K.mean(K.square(K.log(y_true + 1) - K.log(y_pred + 1)), axis=-1)
    model = tf.keras.models.load_model(model_path, custom_objects={"r_loss": r_loss })
    return model

def to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tf_lite_model = converter.convert()
    return tf_lite_model

def to_quant(model):
    converter = tf.lite.TFLiteConverter.from_keraas_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")

def main(args):
    if(args.convert_tflite):
       check_dir(args.tflite_model)
       tflite_temp = to_tflite(load_model(args.model))
       open(args.tflite_model+"/tflite_model.tflite", "wb").write(tflite_temp)
    else:
       check_dir(args.quant_model)
       tfquant_temp = to_quant(load_model(args.model))
       open(args.quant_model+"/tquant_model.tflite", "wb").write(tfquant_temp)
       
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantize a model, run inference, and display an image')
    parser.add_argument('--convert_tflite', action='store_true', help='Only convert the model to TFLite')
    parser.add_argument('--convert_quant', action='store_true', help='Only convert the model to quantization')
    parser.add_argument('--model', type=str, required=True, help='Path to the Keras model file')
    parser.add_argument('--tflite_model', type=str, required=False, help='Path to save the TFLite model')
    parser.add_argument('--quant_model', type=str, required=False, help='Path to save the quantized TFLite model')

    args = parser.parse_args()
    main(args)