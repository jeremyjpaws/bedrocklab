import json
import boto3
import uuid
from PIL import Image
from io import BytesIO
from base64 import b64encode
from base64 import b64decode
from IPython.display import display

bedrock = boto3.client(
    service_name='bedrock-runtime'
)

modelId = 'amazon.titan-image-generator-v1'
accept = 'application/json'
contentType = 'application/json'
prompt = """
Donut drinking a cup of coffee in a Star battle
"""
input = {
    'textToImageParams': {'text': prompt},
    'taskType': 'TEXT_IMAGE',
    'imageGenerationConfig': {
        'cfgScale': 9,
        'seed': 0,
        'quality': 'standard',
        'width': 1173,
        'height': 640,
        'numberOfImages':3
    }
}
body = json.dumps(input) 
response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
response = json.loads(response.get('body').read())
images = response.get('images')
for image in images:
    current = Image.open(BytesIO(b64decode(image)))
    filename = str(uuid.uuid4()) + '.png'
    current.save(filename)
    display(Image.open(filename))