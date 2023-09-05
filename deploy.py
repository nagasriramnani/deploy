from flask import Flask, request, jsonify
from serpapi import GoogleSearch
#from pyngrok import ngrok
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model('best_model.h5')
class_names = ['Accessory Gift Set', 'Baby Dolls', 'Backpacks', 'Bangle', 'Basketballs', 'Bath Robe', 'Beauty Accessory', 'Belts', 'Blazers', 'Body Lotion', 'Body Wash and Scrub', 'Booties', 'Boxers', 'Bra', 'Bracelet', 'Briefs', 'Camisoles', 'Capris', 'Caps', 'Casual Shoes', 'Churidar', 'Clothing Set', 'Clutches', 'Compact', 'Concealer', 'Cufflinks', 'Cushion Covers', 'Deodorant', 'Dresses', 'Duffel Bag', 'Dupatta', 'Earrings', 'Eye Cream', 'Eyeshadow', 'Face Moisturisers', 'Face Scrub and Exfoliator', 'Face Serum and Gel', 'Face Wash and Cleanser', 'Flats', 'Flip Flops', 'Footballs', 'Formal Shoes', 'Foundation and Primer', 'Fragrance Gift Set', 'Free Gifts', 'Gloves', 'Hair Accessory', 'Hair Colour', 'Handbags', 'Hat', 'Headband', 'Heels', 'Highlighter and Blush', 'Innerwear Vests', 'Ipad', 'Jackets', 'Jeans', 'Jeggings', 'Jewellery Set', 'Jumpsuit', 'Kajal and Eyeliner', 'Key chain', 'Kurta Sets', 'Kurtas', 'Kurtis', 'Laptop Bag', 'Leggings', 'Lehenga Choli', 'Lip Care', 'Lip Gloss', 'Lip Liner', 'Lip Plumper', 'Lipstick', 'Lounge Pants', 'Lounge Shorts', 'Lounge Tshirts', 'Makeup Remover', 'Mascara', 'Mask and Peel', 'Mens Grooming Kit', 'Messenger Bag', 'Mobile Pouch', 'Mufflers', 'Nail Essentials', 'Nail Polish', 'Necklace and Chains', 'Nehru Jackets', 'Night suits', 'Nightdress', 'Patiala', 'Pendant', 'Perfume and Body Mist', 'Rain Jacket', 'Rain Trousers', 'Ring', 'Robe', 'Rompers', 'Rucksacks', 'Salwar', 'Salwar and Dupatta', 'Sandals', 'Sarees', 'Scarves', 'Shapewear', 'Shirts', 'Shoe Accessories', 'Shoe Laces', 'Shorts', 'Shrug', 'Skirts', 'Socks', 'Sports Sandals', 'Sports Shoes', 'Stockings', 'Stoles', 'Sunglasses', 'Sunscreen', 'Suspenders', 'Sweaters', 'Sweatshirts', 'Swimwear', 'Tablet Sleeve', 'Ties', 'Ties and Cufflinks', 'Tights', 'Toner', 'Tops', 'Track Pants', 'Tracksuits', 'Travel Accessory', 'Trolley Bag', 'Trousers', 'Trunk', 'Tshirts', 'Tunics', 'Umbrellas', 'Waist Pouch', 'Waistcoat', 'Wallets', 'Watches', 'Water Bottle', 'Wristbands']
  # List of class names

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    img_bytes = image.read()


    img = tf.image.decode_png(img_bytes, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = np.expand_dims(img/255, axis=0)

    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]



    q=predicted_class

    params = {
      "q": q,  	# search query
      "tbm": "shop",  # shop results
      "num": 12,
      "gl": "uk",
      "hl": "en",
      "device":"mobile",
      "engine":"google_shopping",
      "api_key": "657fe8cc427776105c2828f44e05c1a918bdd6d02bf4881a3457644ed77995a9"

    }

    search = GoogleSearch(params)

    results = search.get_dict()


    extracted_results = []

    for item in results["shopping_results"]:
        extracted_item = {

            "link": item.get("link", ""),
            "title": item.get("title", ""),
            "price": item.get("price", "")
        }
        extracted_results.append(extracted_item)

    # Printing the extracted results
    for extracted_item in extracted_results:
        #print("Thumbnail:", extracted_item["thumbnail"])
        print("Link:", extracted_item["link"])
        print("Title:", extracted_item["title"])
        print("Price:", extracted_item["price"])
        print("\n")




       # Create a dictionary containing both pieces of information
    response_data = {
        "predicted_class": predicted_class,
        "extracted_results": extracted_results
    }


    return jsonify(response_data)



#public_url = ngrok.connect(addr='5000')
#print('Public URL:', public_url)

if __name__ == '_main_':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
