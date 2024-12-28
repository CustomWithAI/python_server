import base64

# Replace 'path_to_image.jpg' with the path to your ground truth image
with open('./response.jpeg', 'rb') as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

print(base64_image)
