import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model("image_identifier_model.h5")

img = image.load_img("test.jpg", target_size=(64,64))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
class_index = np.argmax(prediction)

print("Predicted Class:", class_index)
