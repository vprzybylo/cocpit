"""
classifies good ice images:
transforms, makes predictions, and appends classification to dataframe 
"""
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict
import os
import cv2
from twilio.rest import Client

class TestDataSet(Dataset):
    def __init__(self, open_dir, file_list):
        
        self.desired_size = 1000
        self.open_dir = open_dir
        self.file_list = list(file_list)
        self.transform = transforms.Compose([
            transforms.Resize((224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.open_dir, self.file_list[idx])
       
        #image = Image.open(img_path)
        
        #images were resized to 1000x1000 initially
        image = cv2.cvtColor(cv2.imread(self.open_dir+self.file_list[idx],
                                        cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.desired_size, self.desired_size),
                           interpolation = cv2.INTER_AREA)

        image = Image.fromarray(image) #convert back to PIL for transforms
        image = image.convert('RGB')
        image = self.transform(image)

        path = self.file_list[idx]
        return (image, path)


def predict(test_loader, class_names, model, device):
    
    ''' Predict the classes of an image using a trained deep learning model.
    '''

#     print(torch.cuda.device_count())
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model)
#     model = model.to(device)
#     model = model.cuda()
#     model.eval()
    
    d = defaultdict(list)
    top_class = []
    all_outputs = []
    for batch_idx, (inputs, img_paths) in enumerate(test_loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            logits = model.forward(inputs)
            ps = F.softmax(logits,dim=1)
            outputs = ps.cpu().numpy()*100 #(batch size, # classes)
            
            all_outputs.append(outputs)
    
            for pred in outputs: #batch
                for c in range(len(class_names)): #class
                    d[class_names[c]].append(pred[c])
                top_class.append(class_names[np.argmax(pred)])
            
    return d, top_class

def send_message():    
    account_sid = "AC6034e88973d880bf2244f62eec6fe356"
    auth_token = 'f374de1a9245649ef5c8bc3f6e4faa97'
    client = Client(account_sid, auth_token)    
    message = client.messages .create(body =  "ML predictions completed!", 
                                      from_ = "+19285175160", #Provided phone number 
                                      to =    "+15187969534") #Your phone number
    message.sid
        

def main(df, open_dir, device, class_names, model):
    pd.options.mode.chained_assignment = None  # default='warn'

    testdata = TestDataSet(open_dir, file_list = df['filename'])

    test_loader = torch.utils.data.DataLoader(testdata, batch_size=100, shuffle=False, 
                               num_workers=20, drop_last=False)
    
    d, top_class = predict(test_loader, class_names, model, device)

    for column in sorted(d.keys()):
        df[column] = d[column]
    df['classification'] = top_class
    send_message();
    return df                   
