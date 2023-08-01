from flask import Flask, request, Response, jsonify
from img_classification.read_image import read_image_from_url
from img_classification.image import predict_category

app = Flask(__name__)

@app.route('/categorize_panorama', methods=['GET'])
def categorize_panorama():
    image_url = request.args.get('url')
    if image_url:
        image = read_image_from_url(image_url)
        if image is not None:
            predicted_label, confidence = predict_category(image)
            return jsonify({
                'predicted_label': predicted_label,
                'confidence': confidence
            })
        else:
            return Response("Error fetching the image", status=500)

if __name__ == '__main__':
    app.run()