"""
classifies good ice images:
transforms, makes predictions, and appends classification to dataframe 
"""
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd

def process_image(image):

    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image

def predict2(path, model, device, topk=9):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    img = Image.open(path)
    img = img.convert('RGB')
    img = process_image(img)
    
    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img)
    
    model.eval()
    inputs = Variable(img).to(device)
    logits = model.forward(inputs)
    
    ps = F.softmax(logits,dim=1)
    topk = ps.cpu().topk(topk)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)

def main(df, open_dir, device, class_names, model):
    pd.options.mode.chained_assignment = None  # default='warn'

    classifications = []
    for file in df['filename']: 
        img_path = open_dir+file
        probs, classes = predict2(img_path, model, device)  
        crystal_names = [class_names[e] for e in classes]
        classifications.append(crystal_names[0])
    print(len(df), len(classifications))
    df['classification'] = classifications
        
    return df