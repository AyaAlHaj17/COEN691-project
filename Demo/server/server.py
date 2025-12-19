import io
from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from pix2pix_gan.pix2pix_gan import load_pix2pix_gan, normalize
from dncnn.dncnn import load_dnccnn
from flask_cors import CORS

#generator, discriminator, checkpoint = load_pix2pix_gan()
model, device = load_dnccnn()

@torch.no_grad()
def predict_dnccnn(model, device, image):
    # Transform used in training
    inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),      # -> [0,1], shape (3,H,W)
    ])

    # To tensor [0,1]
    degraded = inference_transform(image)          # (3,256,256)
    degraded = degraded.unsqueeze(0).to(device)  # (1,3,256,256)

    # Run model
    restored = model(degraded)                   # (1,3,256,256), in [0,1]
    restored = restored.squeeze(0).cpu()         # (3,256,256)

    # Back to PIL
    restored_np = restored.numpy().transpose(1, 2, 0)  # (H,W,3)
    restored_np = np.clip(restored_np, 0.0, 1.0)
    restored_uint8 = (restored_np * 255).astype(np.uint8)
    restored_img = Image.fromarray(restored_uint8)

    return restored_img


def predict_pix2pix_gan(generator, image):
    image = normalize(image)
    output = generator(image, training=True) * 0.5 + 0.5
    output = np.clip(output, 0.0, 1.0)
    restored_array = (output * 255).astype(np.uint8)
    return restored_array
    
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/restore", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')


    #image_array = np.array(image)
    
    #model = request.form.get('model', 'pix2pixgan')

    #pred = predict_pix2pix_gan(generator, tf.expand_dims(image_array.astype(np.float32), axis=0))
    
    pred = predict_dnccnn(model, device, image)

    restored_iamge = pred #Image.fromarray(pred[0])

    # Save to bytes for response
    img_byte_arr = io.BytesIO()
    #print(type(pred))
    restored_iamge.save(img_byte_arr, format='PNG')
    
    img_byte_arr.seek(0)

    print(restored_iamge)
    return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name='restored.png')
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3100)


