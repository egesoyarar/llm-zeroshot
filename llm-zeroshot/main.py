from transformers import pipeline
import yaml
import json

def get_prediction(classifier, message , label):
    results = classifier(message, label)

    label, scores = results['labels'], results['scores']
    print(f"{role} said: '{message}'")
    print(f"The label specific results, {label}: '{scores}'")

with open("config/config.yaml", "r") as f:
    config = yaml.full_load(f)

classifier = pipeline(config["task"],
                      model=config["model"])

data = []
with open(config["data_path"], "r") as f:
    data = json.load(f)

for conversation in data:
    role = conversation["role"]
    message = conversation["message"]

    sentiment = config["sentiment"]
    intent = config["intents"][role]

    labels = [sentiment, intent]
    for label in labels:
        get_prediction(classifier, message, label)    
    