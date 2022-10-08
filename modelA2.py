import torch
import torch.nn as nn
import torch.optim as optim
import pickle
# keep in mind that this will STILL require a BERT model named bert_model to be loaded in for the Collate of the dataloaders to work,
# unless you use different dataloaders (but then the parameters of the LSTM probably need to be changed too)

class ImageCaptionClassifier(nn.Module):
    # class of the model itself; while initializing it needs to be told the desired size of the hidden layer as well as the
    # dimensions of the RGB images it will be fed
    def __init__(self, hidden_size, height, width):
        super(ImageCaptionClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.height = height
        self.width = width
        
        # defining the three components of the model: layers that will process the image, a layer that will process text, and
        # a layer to process the combined outputs of the previous two
        
        # the image processing layers are mostly taken from Demo 1
        self.layers_image = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=2), 
            nn.BatchNorm2d(3),
            nn.ReLU(),  # I decided to keep these (this and MaxPool2d) in as they will reduce the size of the output
            nn.MaxPool2d(2, 2),
            nn.Tanh()
        )
        
        # the text processing layer is a bidirectional LSTM; the input size here is the size of BERT embeddings, and the 
        # hidden size is tailored to, in the end, be comparable in size to the output from the image layers
        self.layers_text = nn.LSTM(input_size=768, hidden_size=4000, num_layers=1, batch_first=True, bidirectional=True)
        
        # the classification layers are linear layers of decreasing size with different activation functions between them,
        # concluding with reducing the output to 1 and feeding it to a Sigmoid function for a "probability" spread of 0 to 1
        self.layers_classification = nn.Sequential(
            nn.Linear(int(((self.height/2+1)*(self.width/2+1)*3)+8000), self.hidden_size),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, int(self.hidden_size / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.hidden_size / 2), int(self.hidden_size / 4)),
            nn.Tanh(),
            nn.Linear(int(self.hidden_size / 4), 1),
            nn.Sigmoid()            
        )
        
    def forward(self, images, bert_embeddings, device):
        # for calling the model we need to input the images and BERT embeddings from the batch
        # since I have had issues with tensors being 64bit FloatTensors, and not 32bit ones, this part is essential in making
        # it work and it relies on whether it runs on CPU or one of the GPUs
        # the images batches are also restructured to fit what the convolutional layer expects, as per Demo 1
        #if self.device == 'cpu':
        converted_images = images.permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)  
        converted_embeddings = bert_embeddings.type(torch.FloatTensor).to(device)
        #else:
            #print('aha!')
            #converted_images = images.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor).to(self.device)  
            #converted_embeddings = bert_embeddings.type(torch.cuda.FloatTensor).to(self.device)
        
        # the images are fed through the image layers and then reshaped to have fewer dimensions
        processed_images = self.layers_image(converted_images)
        flattened_images = processed_images.reshape(-1, int((self.height/2+1)*(self.width/2+1)*3))
        
        # the captions are fed through the LSTM, and the final hidden states from both directions are combined to form a 
        # sentence representation
        timestep_representation, (final_hidden, final_cell) = self.layers_text(converted_embeddings)
        processed_embeddings = torch.cat((final_hidden[0, :, :], final_hidden[1, :, :]), dim=1)
        
        # data from both of the above is combined and fed to the classification layer
        combined_data = torch.cat((flattened_images, processed_embeddings), dim=1)
        output = self.layers_classification(combined_data)
        
        return output
        
def train(loaders, device, hidden_size=7000, height=100, width=100, model=None):
    # this function trains a given model (or, but default, an image caption classifier model)
    if not model:
        m = ImageCaptionClassifier(hidden_size, height, width).to(device)
    else:
        m = model.to(device)
    
    # BCELoss is used as this is a binary classification problem
    loss = nn.BCELoss()
    optimizer = optim.Adam(m.parameters(), lr=0.00005)
    
    for i in range(0, len(loaders)):
        # len(loaders) is the number of epochs that we set while generating the loaders; it'd be easier to set the number of
        # epochs had this function not had to be in a separate Python file, where I cannot make it dependent on so much
        # of the data processing steps; I would just call the dataloader here separately for every epoch. one of the problems
        # is that the dataloader has to use BERT for embeddings too.
        loader = loaders[i]
        tot_loss = 0
        for j, batch in enumerate(loader):
            # iterating over the loader batch by batch,
            o = m(batch[0], batch[1], device)
            l = loss(o, batch[2].type(torch.FloatTensor).to(device))
            tot_loss += l
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if j%25==0:
                print('\tStill training...')
                
        print("Total loss in epoch {} is {}.".format(i, tot_loss))

    return m

def save_model(model, file_name):
    # a small function intended to be used to save a trained model
    pickle.dump(model, open(file_name, 'wb'))
        
def load_model(file_name):
    # a small function intended to be used to load a trained model
    model = pickle.load(open(file_name, 'rb'))
    
    return model

def test_model(model, loaders, device):
    # this function puts a given model in eval mode and then runs it over "one epoch" of the test data, saving, as lists,
    # the predictions and true classes.
    model.to(device)
    model.eval()  # setting model in eval mode
    all_predictions = []
    all_classes = []
    # this with-statement is recommended for further making sure that the model is not learning from the test data 
    with torch.no_grad():
        for i in range(0, len(loaders)):
            loader = loaders[i]
            for j, batch in enumerate(loader):
                a = batch[0].type(torch.FloatTensor).to(device)
                b = batch[1].type(torch.FloatTensor).to(device)
                
                o = model(a, b, device)
                predictions = torch.squeeze(o).tolist()  # making sure that we get predictions in the correct format
                # encoding the predictions to reflect not probabilities, but classes; anything above 0.5 probability is
                # considered to be a matching image-caption pair (class 1), and below to not be considered that (class 0).
                for i in range(0, len(predictions)):
                    if predictions[i] > 0.5:
                        predictions[i] = 1
                    else:
                        predictions[i] = 0
                all_predictions += predictions
                all_classes += torch.squeeze(batch[2]).tolist()
    
    print('Testing complete!')
        
    return all_predictions, all_classes

def measures(all_predictions, all_classes):
    # this function goes over the output of the testing function and provides some basic evaluation measures, such as
    # accuracy, recall, precision, and f1. 
    print('The following measures have been recorded for this model:')
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    # counting up the true and false positives and negatives
    for i in range(0, len(all_predictions)):
        if all_predictions[i] == all_classes[i]:
            if all_predictions[i] == 1:
                tp += 1
            else:  # if == 0
                tn += 1
        else:  # if not the same
            if all_predictions[i] == 1:
                fp += 1
            else:
                fn += 1
    
    # accuracy can always be calculated
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print(f'\tAccuracy = {accuracy}')
    
    # the following measures cannot always be calculated, so certain conditions are checked for first
    if tp == 0 and fn == 0:
        print('No true positives or false negatives have been recorded, impossible to calculate recall!')
        zero_div = True
    else:
        recall = tp / (tp + fn)
        print(f'\tRecall = {recall}')
    if tp == 0 and fp == 0:
        print('No true or false positives have been recorded, impossible to calculate precision!')
        zero_div = True
    else:
        precision = tp / (tp + fp)
        print(f'\tPrecision = {precision}')
    
    if 'zero_div' in globals():  # I guess I could have checked if precision and recall exist here also
        print('Impossible to calculate f1!')
    else:
        f1 = (2 * recall * precision) / (recall + precision)
        print(f'\tF1 = {f1}')  




