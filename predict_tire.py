"""
TIRE QUALITY PREDICTION SCRIPT
Load trained SE-MobileNetV2 model and predict tire quality from image

Usage:
    python predict_tire.py

Author: Sar-Hal
Date: 2025-01-24
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")


# ==================== SE BLOCK (REQUIRED FOR LOADING MODEL) ====================
class SEBlock(layers.Layer):
    """
    Squeeze-and-Excitation Block
    Must be defined to load the custom model
    """
    def __init__(self, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction = reduction
        
    def build(self, input_shape):
        channels = input_shape[-1]
        self.squeeze = layers.GlobalAveragePooling2D()
        self.excitation = keras.Sequential([
            layers.Dense(channels // self.reduction, activation='relu', use_bias=False),
            layers.Dense(channels, activation='sigmoid', use_bias=False)
        ])
        
    def call(self, inputs):
        # Squeeze: Global spatial information
        squeeze = self.squeeze(inputs)
        
        # Excitation: Channel-wise weights
        excitation = self.excitation(squeeze)
        excitation = tf.reshape(excitation, [-1, 1, 1, tf.shape(inputs)[-1]])
        
        # Scale: Recalibrate channels
        return inputs * excitation
    
    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction})
        return config


# ==================== CONFIGURATION ====================
class Config:
    IMG_SIZE = 224
    CLASS_NAMES = ['defective', 'good']
    MODEL_PATH = 'tire_model_final'  # Folder extracted from zip
    IMAGE_PATH = 'imagebad.png'  # Your test image


# ==================== LOAD MODEL ====================
def load_model():
    """Load the trained model"""
    print("\n" + "="*70)
    print("LOADING TRAINED MODEL")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(Config.MODEL_PATH):
        print(f"\n❌ ERROR: Model folder not found at '{Config.MODEL_PATH}'")
        print("\n📋 Instructions:")
        print("1. Extract 'tire_model_final.zip' to get the model folder")
        print("2. Make sure the folder is in the same directory as this script")
        print("3. The folder should be named 'tire_model_final'")
        return None
    
    try:
        # ✅ FIX FOR KERAS 3: Use TFSMLayer wrapper for SavedModel format
        print("🔧 Detected Keras 3 - Using TFSMLayer wrapper...")
        
        # Load the SavedModel using TFSMLayer
        tfsm_layer = keras.layers.TFSMLayer(
            Config.MODEL_PATH, 
            call_endpoint='serving_default'
        )
        
        # Create a functional wrapper to make it behave like a normal model
        inputs = keras.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3))
        
        # The TFSMLayer returns a dictionary, we need to extract the output
        outputs_dict = tfsm_layer(inputs)
        
        # Extract the actual prediction tensor
        # Try common output keys
        if isinstance(outputs_dict, dict):
            # Try to find the output tensor
            possible_keys = ['output_0', 'dense_1', 'predictions', 'output']
            output_tensor = None
            
            print(f"  Available output keys: {list(outputs_dict.keys())}")
            
            for key in outputs_dict.keys():
                output_tensor = outputs_dict[key]
                break
        else:
            output_tensor = outputs_dict
        
        # Create the model
        model = keras.Model(inputs=inputs, outputs=output_tensor)
        
        print(f"✓ Model loaded successfully from: {Config.MODEL_PATH}")
        print(f"✓ Model input shape: {model.input_shape}")
        print(f"✓ Model output shape: {model.output_shape}")
        
        return model
    
    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        print(f"\n🔍 Debug info: {type(e).__name__}")
        
        # Try alternative loading method
        try:
            print("\n🔧 Trying alternative loading method...")
            import tensorflow as tf
            
            # Load as TensorFlow SavedModel directly
            loaded = tf.saved_model.load(Config.MODEL_PATH)
            
            # Get the inference function
            infer = loaded.signatures['serving_default']
            
            print("✓ Loaded as TensorFlow SavedModel")
            print(f"  Available signatures: {list(loaded.signatures.keys())}")
            
            # Create a wrapper function
            class ModelWrapper:
                def __init__(self, infer_func):
                    self.infer = infer_func
                    self.input_shape = (None, Config.IMG_SIZE, Config.IMG_SIZE, 3)
                    self.output_shape = (None, 2)
                
                def predict(self, x, verbose=0):
                    # Convert input to tensor
                    if not isinstance(x, tf.Tensor):
                        x = tf.constant(x, dtype=tf.float32)
                    
                    # Get prediction
                    result = self.infer(x)
                    
                    # Extract the output (it's a dictionary)
                    if isinstance(result, dict):
                        # Get the first (and likely only) output
                        output_key = list(result.keys())[0]
                        predictions = result[output_key].numpy()
                    else:
                        predictions = result.numpy()
                    
                    return predictions
            
            wrapper = ModelWrapper(infer)
            print("✓ Created model wrapper")
            
            return wrapper
            
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")
            return None


# ==================== PREPROCESS IMAGE ====================
def preprocess_image(image_path):
    """Load and preprocess image for prediction"""
    print("\n" + "="*70)
    print("PREPROCESSING IMAGE")
    print("="*70)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\n❌ ERROR: Image not found at '{image_path}'")
        print("\n📋 Instructions:")
        print("1. Make sure 'image.png' is in the same directory as this script")
        print("2. Or update Config.IMAGE_PATH with the correct path")
        return None, None
    
    try:
        # Load image
        img = Image.open(image_path)
        original_img = img.copy()
        
        print(f"✓ Image loaded: {image_path}")
        print(f"  Original size: {img.size}")
        print(f"  Mode: {img.mode}")
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print(f"  Converted to RGB")
        
        # Resize to model input size
        img = img.resize((Config.IMG_SIZE, Config.IMG_SIZE))
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        
        print(f"  Preprocessed shape: {img_batch.shape}")
        
        return img_batch, original_img
    
    except Exception as e:
        print(f"\n❌ ERROR preprocessing image: {e}")
        return None, None


# ==================== PREDICT ====================
def predict(model, image_path):
    """Make prediction on image"""
    print("\n" + "="*70)
    print("MAKING PREDICTION")
    print("="*70)
    
    # Preprocess image
    img_batch, original_img = preprocess_image(image_path)
    
    if img_batch is None:
        return
    
    # Get prediction
    predictions = model.predict(img_batch, verbose=0)
    
    # Extract probabilities
    prob_defective = predictions[0][0] * 100
    prob_good = predictions[0][1] * 100
    
    # Get predicted class
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = Config.CLASS_NAMES[predicted_class_idx]
    confidence = np.max(predictions[0]) * 100
    
    # Print results
    print(f"\n{'='*70}")
    print("PREDICTION RESULTS")
    print(f"{'='*70}")
    print(f"\n🔍 Image: {image_path}")
    print(f"\n📊 Class Probabilities:")
    print(f"   Defective: {prob_defective:.2f}%")
    print(f"   Good:      {prob_good:.2f}%")
    print(f"\n🎯 Predicted Class: {predicted_class.upper()}")
    print(f"💪 Confidence: {confidence:.2f}%")
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    if predicted_class == 'good':
        print("✅ The tire appears to be in GOOD condition")
        print("   No significant defects detected")
    else:
        print("⚠️  The tire appears to be DEFECTIVE")
        print("   Potential defects or damage detected")
    
    if confidence >= 90:
        print(f"\n🔒 High confidence prediction ({confidence:.1f}%)")
    elif confidence >= 70:
        print(f"\n⚡ Moderate confidence prediction ({confidence:.1f}%)")
    else:
        print(f"\n⚠️  Low confidence prediction ({confidence:.1f}%)")
        print("   Consider manual inspection")
    
    print(f"{'='*70}\n")
    
    # Visualize result
    visualize_prediction(original_img, predicted_class, confidence, 
                        prob_defective, prob_good, image_path)
    
    return predicted_class, confidence, predictions[0]


# ==================== VISUALIZATION ====================
def visualize_prediction(image, predicted_class, confidence, 
                        prob_defective, prob_good, image_path):
    """Visualize the prediction with the original image"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Original image with prediction
    axes[0].imshow(image)
    axes[0].axis('off')
    
    color = 'green' if predicted_class == 'good' else 'red'
    title = f"Predicted: {predicted_class.upper()}\nConfidence: {confidence:.2f}%"
    axes[0].set_title(title, fontsize=14, fontweight='bold', color=color, pad=15)
    
    # Right: Probability bar chart
    classes = Config.CLASS_NAMES
    probabilities = [prob_defective, prob_good]
    colors = ['#ff6b6b', '#51cf66']
    
    bars = axes[1].barh(classes, probabilities, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        axes[1].text(prob + 2, i, f'{prob:.2f}%', 
                    va='center', fontsize=12, fontweight='bold')
    
    axes[1].set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Class Probabilities', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlim(0, 110)
    axes[1].grid(axis='x', alpha=0.3, linestyle='--')
    
    # Highlight predicted class
    predicted_idx = 0 if predicted_class == 'defective' else 1
    axes[1].get_yticklabels()[predicted_idx].set_fontweight('bold')
    axes[1].get_yticklabels()[predicted_idx].set_fontsize(13)
    
    plt.suptitle(f'Tire Quality Prediction: {os.path.basename(image_path)}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save result
    output_filename = 'prediction_result.png'
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    print(f"📊 Visualization saved as: {output_filename}")
    
    plt.show()


# ==================== BATCH PREDICTION ====================
def predict_batch(model, image_folder='test_images'):
    """Predict on multiple images in a folder"""
    print("\n" + "="*70)
    print("BATCH PREDICTION MODE")
    print("="*70)
    
    if not os.path.exists(image_folder):
        print(f"\n⚠️  Folder not found: {image_folder}")
        return
    
    # Get all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_folder, ext)))
    
    if len(image_files) == 0:
        print(f"\n⚠️  No images found in {image_folder}")
        return
    
    print(f"\nFound {len(image_files)} images")
    print("\nProcessing...")
    
    results = []
    
    for img_path in image_files:
        print(f"\n{'='*50}")
        print(f"Image: {os.path.basename(img_path)}")
        
        # Preprocess
        img_batch, _ = preprocess_image(img_path)
        if img_batch is None:
            continue
        
        # Predict
        predictions = model.predict(img_batch, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = Config.CLASS_NAMES[predicted_class_idx]
        confidence = np.max(predictions[0]) * 100
        
        print(f"Prediction: {predicted_class.upper()} ({confidence:.2f}%)")
        
        results.append({
            'image': os.path.basename(img_path),
            'prediction': predicted_class,
            'confidence': confidence
        })
    
    # Summary
    print("\n" + "="*70)
    print("BATCH PREDICTION SUMMARY")
    print("="*70)
    
    for result in results:
        status_icon = "✅" if result['prediction'] == 'good' else "⚠️"
        print(f"{status_icon} {result['image']:<30} -> {result['prediction']:<12} ({result['confidence']:.2f}%)")
    
    # Statistics
    good_count = sum(1 for r in results if r['prediction'] == 'good')
    defective_count = len(results) - good_count
    
    print(f"\n📊 Statistics:")
    print(f"   Total images: {len(results)}")
    print(f"   Good tires: {good_count} ({good_count/len(results)*100:.1f}%)")
    print(f"   Defective tires: {defective_count} ({defective_count/len(results)*100:.1f}%)")


# ==================== MAIN ====================
def main():
    """Main prediction pipeline"""
    print("\n" + "="*70)
    print("  TIRE QUALITY PREDICTION")
    print("  SE-MobileNetV2 Model")
    print("="*70)
    print(f"\nAuthor: Sar-Hal")
    print(f"Date: 2025-01-24\n")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Check if image exists
    if os.path.exists(Config.IMAGE_PATH):
        # Predict on single image
        predicted_class, confidence, probabilities = predict(model, Config.IMAGE_PATH)
    else:
        print(f"\n⚠️  Image not found: {Config.IMAGE_PATH}")
        print("\n📋 Options:")
        print("1. Place 'image.png' in the current directory")
        print("2. Update Config.IMAGE_PATH in the script")
        print("3. Use batch prediction mode for multiple images")
        
        # Ask if user wants to try batch mode
        batch_folder = 'test_images'
        if os.path.exists(batch_folder):
            print(f"\n💡 Found '{batch_folder}' folder. Run batch prediction? (y/n): ", end='')
            # Note: In Kaggle/notebook environment, you'd need to handle input differently
    
    print("\n✅ Prediction complete!")


# ==================== RUN ====================
if __name__ == "__main__":
    from glob import glob
    main()
    
    print("\n" + "="*70)
    print("📝 USAGE NOTES")
    print("="*70)
    print("""
    Single Image Prediction:
        1. Place your image as 'image.png' in the same folder
        2. Run: python predict_tire.py
    
    Custom Image Path:
        Update Config.IMAGE_PATH in the script
    
    Batch Prediction:
        1. Create a 'test_images' folder
        2. Add multiple tire images
        3. Uncomment batch prediction code in main()
    
    Model Requirements:
        - Extract 'tire_model_final.zip'
        - Folder should be named 'tire_model_final'
        - Place in same directory as script
    """)
    print("="*70)
