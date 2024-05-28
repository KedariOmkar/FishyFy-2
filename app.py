import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

""" Importing the Flask Modules """
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_cors import CORS

""" Importing the Required External Libarires """

import base64
import numpy as np
import requests
from pymongo import MongoClient
from urllib.parse import quote_plus
from PIL import Image
from tensorflow.keras import applications
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from io import BytesIO
import tensorflow as tf
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from inference_sdk import InferenceHTTPClient



""" Creating the instance of the flask application """
app = Flask(__name__)
CORS(app)

# Secret key for session management
app.secret_key = 'elon-musk'

UPLOAD_FOLDER = 'media_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


""" MongoDB Atlas Connection """
username = "OnkIndustries"
password = "Asus15@9527"
cluster_name = "onkcloud.8y9wfls.mongodb.net"
database_name = "fish_Freshness_Prediction"  # Replace with your actual database name
collection_name = "fish_data"  # Replace with your actual collection name
# Escape username and password
escaped_username = quote_plus(username)
escaped_password = quote_plus(password)

# Create a MongoClient using the connection string
connection_string = f"mongodb+srv://{escaped_username}:{escaped_password}@{cluster_name}/{database_name}?retryWrites=true&w=majority"
client = MongoClient(connection_string)
# Access the database
db = client[database_name]
# Access the collection
collection = db[collection_name]


""" Machine Learning Models """


# This models detects the image is fish or not
def ifFishOrNot(image_data):
    # Check if image_data is None
    if image_data is None:
        print("Error: image_data is None.")
        return -1  # Or any other appropriate error code or value

    species_list = ['Gourami', 'Midnight Lightning Clownfish', 'greenling', 'sailfish', 'Rockfish',
                    'bearded fireworm', 'sabertooth', 'leafy seadragon', 'grunt', 'Royal Gramma', 'Wolffish',
                    'Red Emperor Snapper', 'Golden Tilefish', 'pompano', 'Tetra', 'Croaker', 'whiting',
                    'dwarf gourami', 'salmon', 'Viperfish', 'sailfin catfish', 'white catfish', 'rougheye rockfish',
                    'yellowtail', 'Banded Wrasse', 'Golden Dorado', 'wrasse', 'Green Clown Goby', 'Plecostomus',
                    'flounder', 'Black Marlin', 'flathead', 'Sea Robin', 'Chalk Bass', 'redtail catfish',
                    'Candy Basslet', 'Bullhead', 'hammerhead shark', 'cutlassfish', 'Diamond Watchman Goby',
                    'brill', 'bluntnose knifefish', 'zooanthid', 'glass knife fish', 'Indian Threadfish',
                    'rockfish', 'tilefish', 'mantis shrimp', 'jackfish', 'Orangespine Unicornfish', 'Rasbora',
                    'halfbeak', 'snook', 'oyster', 'Lyretail Anthias', 'Sixbar Wrasse', 'Betta Fish', 'Bigeye Scad',
                    'Medusafish', 'Yellowtail', 'Arowana', 'shrimp', 'Mandarinfish', 'zebrapleco', 'Kingfish',
                    'Firefly Squid', 'Parrotfish', 'Celestial Pearl Danio', 'pimelodid catfish', 'lobster',
                    'Rainbow Runner', 'Haddock', 'squid', 'Snapper', 'sea squirt', 'Bicolor Blenny', 'Wahoo',
                    'Longnose Butterflyfish', 'dolly varden trout', 'Purple Tang', 'tilapia', 'Giant Oarfish',
                    'whitefish', 'Piranha', 'sea slug', 'Turbot', 'Snowflake Clownfish', 'skate', 'bluefin tuna',
                    'blowfish', 'Ocellaris Clownfish', "Kaudern's Cardinalfish", 'butterflyfish', 'pike',
                    'snakehead', 'mosquitofish', 'Handfish', 'sea butterfly', 'King Threadfin', 'hagfish',
                    'Butterfish', 'unicornfish', 'Longfin Tuna', 'Archer Catfish', 'Mangrove Snapper', 'arapaima',
                    'sculpin', 'Halibut', 'Cichlid', 'Jewel Damselfish', 'Tiger Shovelnose Catfish',
                    'Frostbite Clownfish', 'Anchovy', 'octopus', 'Zebra Pleco', 'Diamond Goby', 'whiskerfish',
                    'tadpole', 'barramundi', 'Eel', 'searobin', 'Percula Clownfish', 'herring', 'Regal Angelfish',
                    'zebraple', 'Yellow Tang', 'sea spider', 'yellow perch', 'Koran Angelfish', 'manta ray',
                    'sturgeon', 'coral trout', 'Rock Beauty', 'Yellowbelly Flounder', 'Salmon', 'Atlantic Herring',
                    'mahi-mahi', 'triggerfish', 'Leafy Sea Dragon', 'long-whiskered catfish', 'Turquoise Killifish',
                    'Pacific Hake', 'Lined Seahorse', 'cuttlefish', 'Anthias', "Scott's Velvet Fairy Wrasse",
                    'Angelfish', 'sweeper', 'platy', 'stonefish', 'Queen Triggerfish', 'Emperor Angelfish',
                    'whale catfish', 'Blind Goby', 'Tube-eye', 'frilled shark', 'dragonet',
                    'Yellowspotted Trevally', 'zamurito', 'Copperbanded Butterflyfish', 'ling', 'Flame Angelfish',
                    'Glass Knifefish', 'bullhead', 'lamprey', 'Yellowstripe Scad', 'grayling', 'Redfish',
                    'anglerfish', 'zonetail', 'brycon', 'Atlantic Mackerel', "Randall's Goby", 'Zebrafish',
                    'Peacock Flounder', 'Monkfish', 'wolf fish', 'Koi', 'Mahi-mahi', 'cichlid', 'halibut',
                    'warty sea cucumber', 'surgeonfish', 'bichir', "lion's mane jellyfish", 'leaffish',
                    'vampire fish', 'Perch', 'snapper', 'sea lily', 'Barrel-eye', 'Cobia', 'Giant Snakehead',
                    'Scooter Dragonet', 'urchin', 'zebrashark', 'Indian Ocean Sailfin Tang',
                    'Golden Head Sleeper Goby', 'water flea', 'axolotl', 'mola mola', 'Trout', 'Lanternfish',
                    'Corydoras Catfish', 'Leopard Whipray', 'Longfin Escolar', 'Midas Blenny', 'stingray',
                    'Bigfin Reef Squid', 'Redtail Catfish', 'sea cucumber', 'zander', 'starfish', 'frog', 'arawana',
                    'sea anemone', 'Lemonpeel Angelfish', 'Rainbowfish', 'Sardine', 'trevally', 'weeverfish',
                    'mudfish', 'platydoras', 'Blue Tang', 'Stoplight Loosejaw', 'Bicolor Pseudochromis', 'plaice',
                    'Chinook Salmon', 'Dolphin Fish', 'shark', 'Red Sea Sailfin Tang', 'pollock', 'crab',
                    'john dory', 'Wrasse', 'clownfish', 'angel shark', 'Fire Goby', 'ziggies', 'pollack', 'eel',
                    'Bass', 'Barracuda', 'redfish', 'zugzug', 'Albacore Tuna', 'Archerfish', 'Pacific Cod',
                    'Elephantnose Fish', 'Foxface Rabbitfish', 'betta', 'Tilefish', 'anchovy', 'Lawnmower Blenny',
                    'newt', 'pickerel', 'bream', 'Pinecone Fish', 'Spanish Mackerel', 'glass catfish',
                    'Giant Trevally', 'Majestic Angelfish', 'barracuda', 'Pufferfish', 'Fourspot Butterflyfish',
                    'electric catfish', 'Sixline Wrasse', 'Pollock', 'mackerel', 'bristle mouth', 'snail',
                    'talking catfish', 'wolffish', 'barnacle', 'Colossal Squid', 'jawfish', 'tubeblenny',
                    'Yellow Banded Pipefish', 'Neon Dottyback', 'blue tang', 'Bluebanded Goby', 'Threadfin Bream',
                    'Barramundi Fish', 'Grouper', 'Yellowfin Surgeonfish', 'pencil catfish', 'smelt', 'scallop',
                    'chub', 'parrotfish', 'rainbow trout', 'Boarfish', 'Black Ice Ocellaris Clownfish',
                    'Gulper Eel', 'tigerfish', 'Jackfish', 'Bumblebee Goby', 'caecilian', 'knifefish',
                    'Flame Hawkfish', 'Molly', 'flagfish', 'Lingcod', 'blind cavefish', 'gudgeon',
                    'Yasha White Ray Shrimp Goby', 'Herring', 'stickleback', 'lions mane jellyfish', 'pufferfish',
                    'Marlin', 'sea urchin', 'sheatfish', 'zanderfish', 'koi', 'torsk', 'Rainbow Trout',
                    'Fiji Blue Devil Damsel', 'Twinspot Goby', 'Spotfin Hogfish', 'Red Mandarin Dragonet',
                    'pipefish', 'Glassfish', 'Banggai Cardinalfish', 'krill', 'Freckled Hawkfish',
                    'Powder Blue Tang', 'Gulf Menhaden', 'Spotted Grunt', 'zorsefish', 'goliath tigerfish',
                    'swordtail', 'tang', 'Tigerfish', 'Green Chromis', 'lungfish', 'Goldfish', 'Longnose Gar',
                    'bluefish', 'Yellow Eye Kole Tang', 'moonfish', 'Slippery Dick', 'Humboldt Squid', 'seahorse',
                    'Japanese Amberjack', 'Orchid Dottyback', 'Flagfish', 'Pompano', 'bobbit worm', 'Killifish',
                    'boxfish', 'paddlefish', 'Bonito', 'Ornate Leopard Wrasse', 'Threadfin Geophagus',
                    'electric ray', 'Pajama Cardinalfish', 'prawn', 'Red-bellied Piranha', 'guppy', 'lionfish',
                    'zebrafish', 'Stargazer', 'Dusky Grouper', 'Discus', 'Bluestreak Cleaner Wrasse',
                    'piraiba catfish', 'Green Jobfish', 'Dragon Wrasse', 'African Pike', 'Clown Loach',
                    'Yellow Pyramid Butterflyfish', 'moon jellyfish', 'dogfish', 'tinfoil barb', 'Red Fire Goby',
                    'Jellybean Parrotfish', 'Dragonet', 'Yellow Banded Possum Wrasse', 'thresher shark', 'Cod',
                    'Blue Spot Jawfish', 'Triggerfish', 'whitefin wolf fish', 'zorse', 'Clown Knifefish',
                    'Yellowtail Damselfish', 'gurnard', 'threadfin', 'Pike', 'Orange Spotted Goby', 'Bristlemouth',
                    'Orange Roughy', 'Glass Squid', 'Silver Carp', 'marlin', 'Ribbonfish', 'Psychedelic Mandarin',
                    'Shortraker Rockfish', 'Cherubfish', 'arowana', 'Clown Coris Wrasse', 'sea angel', 'Sturgeon',
                    'Silver Arowana', 'tarpon', 'Amberjack', 'Blanket Octopus', 'Betta', 'tigerperch', 'sea fan',
                    'muskellunge', 'striped bass', 'fluke', 'Electric Blue Hap', 'Ribboned Seadragon',
                    'Blueface Angelfish', 'goblin shark', 'zebra oto', 'brittle star', 'leopardfish', 'Neon Tetra',
                    'Golden Snapper', 'glassfish', 'Gulper Shark', 'isopod', 'comb jelly', 'clam', 'Chimaera',
                    'trout', 'Carp', 'monkfish', 'Achilles Tang', 'giant danio', 'Japanese Eel', 'Bluefish',
                    'weedy seadragon', 'Powder Brown Tang', 'Yellow Watchman Goby', 'Darter', 'Batfish', 'mullet',
                    'chambered nautilus', 'piranha', 'Mandarin Goby', 'bluegill', 'Bluestripe Snapper',
                    'soft coral', 'Pink Salmon', 'longfin smelt', 'Shortnose Greeneye', 'Bluefin Trevally',
                    'Bobtail Squid', 'electric eel', 'Moorish Idol', 'coelacanth', 'sawfish', 'slug',
                    'silver dollar fish', 'drumfish', 'catfish', 'Goby', 'scorpionfish', 'killifish', 'swordfish',
                    'Razorfish', 'Yelloweye Rockfish', 'Hooded Fairy Wrasse', 'wahoo', 'Yellowfin Croaker', 'cod',
                    'Red Drum', 'Mocha Storm Clownfish', 'argentine blue-bill', 'amphipod', 'pangasius',
                    'hard coral', 'Flounder', 'Drum', 'sun catfish', 'goby', 'gourami', 'tailspot blenny', 'hake',
                    'squatina', 'Vampire Squid', 'mussel', 'Koi Fish', 'Blackfin Tuna', 'Weedy Sea Dragon',
                    'Tilapia', 'angelfish', 'nudibranch', 'zebraperch', 'walleye', 'Hogfish', 'Horse Mackerel',
                    'Copperband Butterflyfish', 'Vlamingii Tang', 'Tuna', 'Fairy Wrasse', 'carp', 'basket star',
                    'sunfish', 'giraffe catfish', 'Harlequin Tuskfish', 'firefly squid', 'Swordtail', 'ribbon eel',
                    'Bigeye Tuna', 'Barb', 'sheepshead', 'medusafish', 'haddock', 'Swordfish', 'Snakehead',
                    'rainbowfish', 'trumpetfish', 'spookfish', "McCulloch's Clownfish", 'Pineapplefish', 'Platy',
                    'Atlantic Cod', 'Guppy', 'Skate', 'Blue Throat Triggerfish', 'goldfish', 'sardine',
                    'Yellow Coris Wrasse', 'Ghost Pipefish', 'batfish', 'Gurnard', 'zooids', 'Boxfish', 'garfish',
                    'Queen Parrotfish', 'zoogoneticus', 'Blue Hippo Tang', 'perch', 'Melanurus Wrasse',
                    'zebra pleco', 'giant trevally', 'wallago catfish', 'grouper', 'Mackerel', 'snipefish',
                    'Domino Damsel', 'sargo', 'Japanese Swallowtail Angelfish', 'Catfish', 'Tripodfish',
                    'feather star', 'placidochromis', 'Powder Blue Surgeonfish', 'Corydoras', 'tuna', 'zoarcid',
                    'Alaskan Pollock', 'Tube Snout', 'plecostomus', 'hoki', 'spotted talking catfish', 'ribbonfish',
                    'coral', 'Tiger Jawfish', 'sole', 'Paper Nautilus', 'salamander', 'gar', 'Yellow Perch',
                    'turbot', 'Sole', 'Japanese Sea Bass', 'Platinum Percula Clownfish', 'lingcod', 'bowfin',
                    'thornyhead', 'Black Sea Bass', 'Warty Sea Cucumber', 'Danio', "Carpenter's Flasher Wrasse",
                    'filefish', 'Blackear Wrasse', 'Milkfish', 'Mango Clownfish', 'cowfish', 'Blue Catfish',
                    'hogfish', 'platypus']

    # Load the pre-trained InceptionV3 model
    model = InceptionV3(weights='imagenet')

    # Function to predict objects in an image
    def predict_objects(image_data):
        # Decode base64 image data
        try:
            image_data_decoded = base64.b64decode(image_data)
        except Exception as e:
            print(f"Error decoding base64 image data: {e}")
            return -1  # Or any other appropriate error code or value

        # Convert to PIL Image
        image = Image.open(BytesIO(image_data_decoded)).resize((299, 299))  # InceptionV3 input size

        # Convert 4-channel images to 3 channels (RGBA to RGB)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Convert PIL Image to numpy array
        image_array = np.array(image)

        # Expand dimensions and preprocess input
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        # Make a prediction using the pre-trained model
        predictions = model.predict(image_array)

        # Decode and get the top predicted labels
        decoded_predictions = decode_predictions(predictions, top=10)[0]
        top_labels = [label for (_, label, _) in decoded_predictions]

        predicted_species = 0
        for x in top_labels:
            if x in species_list:
                predicted_species += 1

        return predicted_species

    result = predict_objects(image_data)
    return result


# This models predicts the species of the fish
def checkSpecies(model_path, base64_image):
    def predict_with_saved_model(model, image_data):
        # Decode base64 image data
        image_data_decoded = base64.b64decode(image_data)

        # Convert to PIL Image
        img = image.load_img(BytesIO(image_data_decoded), target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted class label
        predicted_class = np.argmax(predictions)

        return predicted_class

    def map_index_classes(index_id):
        map_dict = {
            0: 'Bangus', 1: 'Big Head Carp', 2: 'Black Spotted Barb', 3: 'Catfish', 4: 'Climbing Perch',
            5: 'Fourfinger Threadfin', 6: 'Freshwater Eel', 7: 'Glass Perchlet', 8: 'Goby', 9: 'Gold Fish',
            10: 'Gourami', 11: 'Grass Carp', 12: 'Green Spotted Puffer', 13: 'Indian Carp',
            14: 'Indo-Pacific Tarpon', 15: 'Jaguar Gapote', 16: 'Janitor Fish', 17: 'Knifefish',
            18: 'Long-Snouted Pipefish', 19: 'Mosquito Fish', 20: 'Mudfish', 21: 'Mullet', 22: 'Pangasius',
            23: 'Perch', 24: 'Scat Fish', 25: 'Silver Barb', 26: 'Silver Carp', 27: 'Silver Perch', 28: 'Snakehead',
            29: 'Tenpounder', 30: 'Tilapia'
        }

        class_name = map_dict.get(index_id)

        return class_name

    # Load the model
    model = load_model(model_path)

    # Test the model
    predict = predict_with_saved_model(model, base64_image)
    class_name = map_index_classes(predict)

    return class_name


# This functions convert the image to base_64 format
def image_to_base64(image_path):
    """
    Convert an image to base64 format.

    Args:
    image_path (str): Path to the image file.

    Returns:
    str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string




""" This are the External Machine learning Models """
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="wp0g5gnriLp6aCNPug44"
)


""" Nodemcu routes """
# Nodemcu ip Address
NODEMCU_ENDPOINT_connect_board = "http://192.168.147.213/connect_board"
NODEMCU_ENDPOINT_start_readings = "http://192.168.147.213/start_readings"



""" Routes for the Server """

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/checkFreshness')
def checkFreshness():
    return render_template('checkFreshness.html')

@app.route('/viewSpecies')
def viewSpecies():
    # Retrieve data from MongoDB
    fish_data = collection.find()
    # Pass data to the frontend
    return render_template('viewSpecies.html', fish_data_result=fish_data)

@app.route('/details/<species_name>')
def details(species_name):
    try:
        # Retrieve data from MongoDB based on species_name
        fish_data = collection.find_one({"fish_name": species_name})

        if fish_data:
            # Pass data to the frontend
            return render_template('fishDetails.html', fish_data=fish_data)
        else:
            # Handle case where species_name is not found
            return render_template('error.html', error_message='Fish not found')

    except Exception as e:
        print(f"Error: {e}")



@app.route('/checkSmell')
def checkSmell():
    return render_template('checkSmell.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')


@app.route('/scan', methods=['POST'])
def scan_image():
    print("scan route fired..")
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    image = request.files['image']
    
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the image file
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    print(image_path)
    
    # Here, add your image processing logic
    imagebase64 = image_to_base64(image_path)

    check_fish_result = ifFishOrNot(imagebase64)

    if check_fish_result >= 1:
        prediction = 'fish'
            
        try:
        # Replace this with the actual species name obtained from your model
            species_detected = checkSpecies('./static/models/Species_Model.h5',imagebase64)
            print(species_detected)
        except Exception as e:
            print('Exception',e) 

    else:
        prediction = 'not fish'
        species_detected = 'N/A'

    return jsonify({
        'result': {
            'message': 'Image received and saved successfully',
            'image_path': image_path,
            'check_fish_result': prediction,
            'species_detected': species_detected,
            'image_data': imagebase64  # Add image data to the response
        }
    }), 200


@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        print("Result Route Fired...")
        # Handle the POST request
        species_name = request.form.get('species_name')
        image_path = request.form.get('image_path')

        print("---")
        print("\t",image_path)
        print("\t",species_name)
        print("--")

        model_ids = [
            "fish-disease-t6b03/1",
            "fish-disease-qvxvl/1",
            "fish-health/1",
            "fish-disease-detection-pjsmt/2",
            "diseased-fish/1",
            "fish-disease-detection-n4fho/2",
            "multiple-fish-disease/3",
            "fish_diseases_classification/1",
            "finspy/1"
        ]

        resultList = []

        def infer_model(model_id):
            result = CLIENT.infer(image_path, model_id=model_id)
            if model_id in ["fish-disease-t6b03/1", "fish-health/1", "fish_diseases_classification/1"]:
                return result["top"]
            elif model_id in ["fish-disease-qvxvl/1", "fish-disease-detection-n4fho/2", "multiple-fish-disease/3"]:
                return [prediction["class"] for prediction in result["predictions"]]
            elif model_id in ["fish-disease-detection-pjsmt/2", "finspy/1"]:
                return result["predicted_classes"]
            elif model_id == "diseased-fish/1":
                return [cls for prediction in result["predictions"] for cls in prediction["class"]]

        async def main():
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=len(model_ids)) as executor:
                tasks = [loop.run_in_executor(executor, infer_model, model_id) for model_id in model_ids]
                results = await asyncio.gather(*tasks)

                for result in results:
                    if isinstance(result, list):
                        resultList.extend(result)
                    else:
                        resultList.append(result)

        start_time = time.time()
        asyncio.run(main())
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")

        # Printing Final List
        print(resultList)

        healthyFishCount = 0
        for x in resultList:
            if x == 'Healthy Fish' or x == 'HF' or x == "healthy-fish" or x == "healthy":
                healthyFishCount += 1

        print("Healthy Count:", healthyFishCount)

        diseaseFishCount = 0
        diseaseList = []
        for x in resultList:
            if x == 'Bacterial diseases - Aeromoniasis' or x == "3" or x == "Rotten gills" or x == "gill_disease" or x == "1" or x == "Eye disease" or x == "Bacterial Red disease" or x == "Parasitic diseases" or x == "sick-fish" or x == "EUS" or x == "BRD" or x == "disease" or x == "Bacterial gill disease":
                diseaseFishCount += 1

        print("Disease Count:", diseaseFishCount)
        unique_diseases = list(set(resultList))
        for diseases in unique_diseases:
            if diseases == "BRD" or diseases == "EUS" or diseases == "gill_disease" or diseases == "Rotten gills" or diseases == "BGD" or diseases == "Parasitic diseases" or diseases == "Bacterial diseases" or diseases == "Baterial diseases - Aeromoniasis" or diseases == "Bacterial Red disease" or diseases == "Bacterial gill disease":
                diseaseList.append(diseases)

        print("Diseases:", diseaseList)

        if healthyFishCount >= 7 or diseaseFishCount <= 3:
            finalResult = "Fresh"
        if diseaseFishCount >= 7 or healthyFishCount <= 3:
            finalResult = "Spoiled"

        print("Final Result:", finalResult)


        # Fetch the data
        fetch_data = collection.find_one({'fish_name':species_name})

        print("Opening the results web page")
        return render_template('predictionResults.html',fish_image = image_path,fish_data=fetch_data,freshness_result=finalResult,disease_list=diseaseList)
    else:
        # Handle other HTTP methods
        return 'Method Not Allowed', 405


# Route to establish the connection with the NodeMCU
@app.route('/connect_board', methods=['GET'])
def connect_board():
    print("ConnectBoard route Fired")

    try:
        response = requests.get(NODEMCU_ENDPOINT_connect_board)
        if response.status_code == 200:
            return jsonify({"status":"connected"})
        else:
            return jsonify({"status":"not connected"})
    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e)}), 500


# Route to check the status of the NodeMCU
@app.route('/check_status', methods=['GET'])
def check_status():
    app.logger.info('Route /check_status fired')
    return jsonify({"status": "on"}), 200

# Route to trigger start_readings route on NodeMCU
@app.route('/scan_smell', methods=['GET'])
def scan_smell():
    print("scan Smell Fired")

    try:
        # Trigger the start_readings route on NodeMCU
        response = requests.get(NODEMCU_ENDPOINT_start_readings)
        if response.status_code == 200:
            # Wait for 20 seconds for the sensor data collection
            #time.sleep(20)

            # Now fetch the gas readings from the NodeMCU
            gas_readings_response = requests.get(NODEMCU_ENDPOINT_start_readings)
            if gas_readings_response.status_code == 200:
                # Extract the gas readings from the response
                gas_readings_data = gas_readings_response.json()

                print(gas_readings_data)

                # Forward the gas readings to the frontend
                return jsonify({"status":"success",
                                **gas_readings_data})
            else:
                return jsonify({"status": "error", "message": "Failed to fetch gas readings from NodeMCU"})
        else:
            return jsonify({"status": "error", "message": "Failed to trigger start_readings route on NodeMCU"})
    except Exception as e:
        return jsonify({"status": "error", "message": "An error occurred", "error": str(e)}), 500



@app.route('/iot_connected',methods = ["GET","POST"])
def iot_connected():
    return render_template('iot_connected.html')



""" This are the error handling pages """
# Custom error handler for 404 Not Found
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Custom error handler for 500 Internal Server Error
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# Custom error handler for 403 Forbidden
@app.errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403

# Custom error handler for 400 Bad Request
@app.errorhandler(400)
def bad_request(e):
    return render_template('400.html'), 400



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)