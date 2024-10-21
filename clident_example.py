import requests
import base64
import re
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Function to encode an image to base64
def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def call_vlm(image: str, text: str) -> str:
    """
    """
    
    data = {
        "image_base64": encode_image_to_base64(image), # base64 encoded image,
        "text": text,   
    }
    res = requests.post("http://localhost:8000/generate", json=data)
    
    if res.status_code != 200:
        raise Exception(f"Failed to call VLM: {res.text}")
    
    return res.json()

def extract_points(molmo_output, image_w, image_h):
    all_points = []
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    return all_points

def draw_points(image_path, model_response):
    image = Image.open(image_path)
    image = np.array(image)
    image_h, image_w, _ = image.shape
    points = extract_points(model_response, image_w, image_h)

    plt.imshow(image)
    for point in points:
        plt.plot(point[0], point[1], 'ro')
    plt.show()
    
if __name__ == "__main__":
    image_path = "data/knife.jpg"
    task = "cutting a cucumber"
    # text = "where the robot should grasp the knife to pick it up? Answer with a point on the image."
    text = f"""You are an intelligent system that can specifiy a grasping point for a robot to pick up an object. Given an image of the scene and a task description, you should provide a point on the image where the robot should grasp the object to best accomplish the task. 
    Note there could be multiple valid grasping points for a grasping the object, but you should provide the one that is most suitable for the task.
    The task is: {task}
    You should think carefully about the robots grasp postion that can best accomplish the task.
    Answer with a point on the image."""
    
    res = call_vlm(image_path, text)
    print(res)
    draw_points(image_path, res['response'])