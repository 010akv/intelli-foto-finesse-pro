from torchvision import models
import numpy as np
import cv2
import argparse
import yaml
import torch

# define a dictionary that maps model names to their classes
# inside torchvision
MODELS = {
    "vgg16": models.vgg16(pretrained=True),
    "vgg19": models.vgg19(pretrained=True),
    "inception": models.inception_v3(pretrained=True),
    "densenet": models.densenet121(pretrained=True),
    "resnet50": models.resnet50(pretrained=True)
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='configuration for classifying images')
    args = parser.parse_args()
    return args

def preprocess_image(image, image_size, mean, std):
    # swap the color channels from BGR to RGB, resize it, and scale
    # the pixel values to [0, 1] range
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype("float32") / 255.0
    # subtract ImageNet mean, divide by ImageNet standard deviation,
    # set "channels first" ordering, and add a batch dimension
    image -= mean
    image /= std
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    # return the preprocessed image
    return image

def setup_model(model_name, device):
    model = MODELS[model_name]
    model = model.to(device)
    model.eval()
    return model

def predict(model_name, image_path, image_size, mean, std, labels_file, device, top_n=5, vis=True):
    model = setup_model(model_name, device)
    image = cv2.imread(image_path)
    orig = image.copy()
    image = preprocess_image(image, image_size, mean, std)
    image = torch.from_numpy(image)
    image = image.to(device)
    labels = dict(enumerate(open(labels_file)))
    preds = model(image)
    probabilities = torch.nn.Softmax(dim=-1)(preds)
    sortedProba = torch.argsort(probabilities, dim=-1, descending=True)

    # loop over the predictions and display the rank-5 predictions and
    # corresponding probabilities to our terminal
    
    for (i, idx) in enumerate(sortedProba[0, :top_n]):
        print("{}. {}: {:.2f}%".format(i, labels[idx.item()].strip(),probabilities[0, idx.item()] * 100))
    
    if vis:
        (label, prob) = (labels[probabilities.argmax().item()],
            probabilities.max().item())
        cv2.putText(orig, "Label: {}, {:.2f}%".format(label.strip(), prob * 100),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Classification", orig)
        cv2.waitKey(0)
    
    return sortedProba
        

def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config_file))
    config = config['classify']
    predict(config['model'], config['image_path'], config['image_size'], config['mean'], config['std'], config['labels_file'], config['device'], config['top_n'])
    
if __name__ == '__main__':
    main()
    