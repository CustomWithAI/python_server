import json

# Your JSON string (or load it from a file)
json_data = '''
{
    "preprocess": {
        "resize": {
            "width": 224,
            "height": 224
        },
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    },
    "other_config": {
        "option1": true,
        "option2": "value"
    }
}
'''

# Parse JSON into a Python dictionary
data = json.loads(json_data)

# Accessing nested elements
preprocess = data["preprocess"]

# Print values
print(f"Preprocess: {preprocess}")
