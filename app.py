import tensorflow as tf
import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
# -----------------------------------------------
# 1. Load Pre-trained Model (MobileNetV2 on ImageNet)
# -----------------------------------------------
model = MobileNetV2(weights="imagenet")

# -----------------------------------------------
# 2. Define All Dog Breed Names (Explicit List)
# -----------------------------------------------
dog_breeds = [
    "affenpinscher", "afghan_hound", "airedale", "akita", "alaskan_malamute",
    "american_staffordshire_terrier", "appenzeller", "australian_terrier", 
    "basenji", "basset", "beagle", "bearded_collie", "bernese_mountain_dog", 
    "bichon_frise", "black_and_tan_coonhound", "blenheim_spaniel", "bloodhound",
    "bluetick", "border_collie", "border_terrier", "borzoi", "boston_bull", 
    "bouvier_des_flandres", "boxer", "brabancon_griffon", "briard", "brittany_spaniel",
    "bull_mastiff", "cairn", "cardigan", "chesapeake_bay_retriever", "chihuahua", 
    "chow", "clumber", "cocker_spaniel", "collie", "curly_coated_retriever", 
    "dandie_dinmont", "dhole", "dingo", "doberman", "english_foxhound", 
    "english_setter", "english_springer", "english_toy_spaniel", "eskimo_dog",
    "flat_coated_retriever", "french_bulldog", "german_shepherd", "german_short_haired_pointer",
    "giant_schnauzer", "golden_retriever", "gordon_setter", "great_dane", "great_pyrenees",
    "greater_swiss_mountain_dog", "groenendael", "ibizan_hound", "irish_setter", 
    "irish_terrier", "irish_water_spaniel", "irish_wolfhound", "italian_greyhound", 
    "japanese_spaniel", "keeshond", "kelpie", "kerry_blue_terrier", "komondor",
    "kuvasz", "labrador_retriever", "lakeland_terrier", "leonberg", "lhasa",
    "malamute", "malinois", "maltese_dog", "mexican_hairless", "miniature_pinscher",
    "miniature_poodle", "miniature_schnauzer", "newfoundland", "norfolk_terrier",
    "norwegian_elkhound", "norwich_terrier", "old_english_sheepdog", "otterhound",
    "papillon", "pekinese", "pembroke", "pomeranian", "pug", "redbone",
    "rhodesian_ridgeback", "rottweiler", "saint_bernard", "saluki", "samoyed",
    "schipperke", "scotch_terrier", "scottish_deerhound", "sealyham_terrier",
    "shetland_sheepdog", "shih_tzu", "siberian_husky", "silky_terrier", "soft_coated_wheaten_terrier",
    "staffordshire_bullterrier", "standard_poodle", "standard_schnauzer", "sussex_spaniel",
    "tibetan_mastiff", "tibetan_terrier", "toy_poodle", "toy_terrier", "vizsla",
    "walker_hound", "weimaraner", "welsh_springer_spaniel", "west_highland_white_terrier",
    "whippet", "wire_haired_fox_terrier", "yorkshire_terrier"
]

# -----------------------------------------------
# 3. Function to Predict Dog Breed
# -----------------------------------------------
def predict_breed(img: Image.Image) -> str:
    # Resize image to 224x224 for MobileNetV2
    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Model Prediction
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]

    # Check if Image Contains a Dog
    for pred in decoded_preds:
        label = pred[1].lower()
        if label in dog_breeds:
            breed = label.replace("_", " ").title()  # Format breed name
            return f"This is a dog! Breed detected: {breed}"

    return "This is NOT a dog."

# -----------------------------------------------
# 4. Deploy as Gradio Web App
# -----------------------------------------------
iface = gr.Interface(
    fn=predict_breed,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Dog Breed Classification",
    description="Upload an image to check if it is a dog and detect its breed. All breed names are included in the code.",
)

iface.launch()
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image

# -----------------------------------------------
# 1. Load Pre-trained Model (MobileNetV2 on ImageNet)
# -----------------------------------------------
model = MobileNetV2(weights="imagenet")

# -----------------------------------------------
# 2. Define All Dog Breed Names (Explicit List)
# -----------------------------------------------
dog_breeds = [
    "affenpinscher", "afghan_hound", "airedale", "akita", "alaskan_malamute",
    "american_staffordshire_terrier", "appenzeller", "australian_terrier", 
    "basenji", "basset", "beagle", "bearded_collie", "bernese_mountain_dog", 
    "bichon_frise", "black_and_tan_coonhound", "blenheim_spaniel", "bloodhound",
    "bluetick", "border_collie", "border_terrier", "borzoi", "boston_bull", 
    "bouvier_des_flandres", "boxer", "brabancon_griffon", "briard", "brittany_spaniel",
    "bull_mastiff", "cairn", "cardigan", "chesapeake_bay_retriever", "chihuahua", 
    "chow", "clumber", "cocker_spaniel", "collie", "curly_coated_retriever", 
    "dandie_dinmont", "dhole", "dingo", "doberman", "english_foxhound", 
    "english_setter", "english_springer", "english_toy_spaniel", "eskimo_dog",
    "flat_coated_retriever", "french_bulldog", "german_shepherd", "german_short_haired_pointer",
    "giant_schnauzer", "golden_retriever", "gordon_setter", "great_dane", "great_pyrenees",
    "greater_swiss_mountain_dog", "groenendael", "ibizan_hound", "irish_setter", 
    "irish_terrier", "irish_water_spaniel", "irish_wolfhound", "italian_greyhound", 
    "japanese_spaniel", "keeshond", "kelpie", "kerry_blue_terrier", "komondor",
    "kuvasz", "labrador_retriever", "lakeland_terrier", "leonberg", "lhasa",
    "malamute", "malinois", "maltese_dog", "mexican_hairless", "miniature_pinscher",
    "miniature_poodle", "miniature_schnauzer", "newfoundland", "norfolk_terrier",
    "norwegian_elkhound", "norwich_terrier", "old_english_sheepdog", "otterhound",
    "papillon", "pekinese", "pembroke", "pomeranian", "pug", "redbone",
    "rhodesian_ridgeback", "rottweiler", "saint_bernard", "saluki", "samoyed",
    "schipperke", "scotch_terrier", "scottish_deerhound", "sealyham_terrier",
    "shetland_sheepdog", "shih_tzu", "siberian_husky", "silky_terrier", "soft_coated_wheaten_terrier",
    "staffordshire_bullterrier", "standard_poodle", "standard_schnauzer", "sussex_spaniel",
    "tibetan_mastiff", "tibetan_terrier", "toy_poodle", "toy_terrier", "vizsla",
    "walker_hound", "weimaraner", "welsh_springer_spaniel", "west_highland_white_terrier",
    "whippet", "wire_haired_fox_terrier", "yorkshire_terrier"
]

# -----------------------------------------------
# 3. Function to Predict Dog Breed
# -----------------------------------------------
def predict_breed(img: Image.Image) -> str:
    # Resize image to 224x224 for MobileNetV2
    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Model Prediction
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]

    # Check if Image Contains a Dog
    for pred in decoded_preds:
        label = pred[1].lower()
        if label in dog_breeds:
            breed = label.replace("_", " ").title()  # Format breed name
            return f"This is a dog! Breed detected: {breed}"

    return "This is NOT a dog."

# -----------------------------------------------
# 4. Deploy as Gradio Web App
# -----------------------------------------------
iface = gr.Interface(
    fn=predict_breed,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Dog Breed Classification",
    description="Upload an image to check if it is a dog and detect its breed. All breed names are included in the code.",
)

iface.launch()
