## ASSIGNMENT 1 - IMAGES AND CAPTIONS notes & remarks  
Maria Irena Szawerna, 08.10.2022  
for Machine learning for statistical NLP: Advanced LT2326  

### PART 1 - COLLECTING TRAINING DATA  
To be honest, this was likely the most time-consuming part of the assignment (and I believe it should be worth proportionally more points as a result). In order to get the necessary data, I load in two of the files (instances and captions), and I proceed to get image-caption samples using a number of custom functions. THe last of them (which also makes use of the other ones), create_samples() has an optional argument 'smart', which is my way of at least partially implementing the bonus part of the assignment, more on which later. In the next stage, sample splits are created. Here the amount of data that will be used later is specified, and the image data is retrieved. A split into 70% training data, 15% testing, and 15% valuation was used, although the last of these splits was not really used during the training and testing proper. Next, a custom Dataset class is created for later easier access, and the samples are saved as such. Subsequently, BERT is loaded in, as it will be needed for encoding the captions. **Important!** BERT must be loaded in when training or testing models, although it is used only for encoding. Next, a function that retrieves the BERT embeddings from its penultimate layer's hidden states is defined, and used in a custom Collate class which is then called in the dataloader in captions_dataloader(). In the end, dataloaders for splits with the size of 700 training / 150 testing and 1400 training / 300 testing are created. This is not much data, and likely not enough to get any good performance out of the model, but unfortunately the server constraints make it very time-consuming to work with bigger splits at the moment. Additionally, as outlined in https://stats.stackexchange.com/questions/226672/how-few-training-examples-is-too-few-when-training-a-neural-network, such small sample sizes *can* work, so it is not entirely ludicrous.

### PART 2 - MODELLING  
All of the code used here can be found in the modelA2.py file, but also commented out in the notebook.  
+ ImageCaptionsClassifier() is a class that defines the model itself. When created it required specifying the desired hidden size as well as the dimensions of the images fed to it. Internally it consists of three segments: layers_image, layers_text, and layers_classification. The first segment is intended to process the image data and is, in the most part, taken from Demo 1, although that code had to be updated; I also decided to keep some elements that were left out in the Demo. First the image data is fed to a convolutional network. That is then normalized, fet through a ReLU activation function, pooled using MaxPool2d (to decrease the size of the output) and put through a Tanh function. The second segment, layers_text takes the BERT embeddings from the dataloader (as far as I understand, they are only retrieved at this point, which is why BERT has to be up) and feed them to a bidirectional LSTM; the final hidden states from both directions are combined to form a sentence representation for later use. Finally, the classification layer is a mix of gradually smaller Linear layers (the output is always half the input size, except the last layer) and a variety of activation functions (LeakyReLU, Tanh, Sigmoid at the very end). Because of how large the input to this segment is, I wanted the decrease in size to happen gradually. The final Linear layer drops the size down to 1, and the Sigmoid function transforms that to a probability distribution. So, to reiterate the possible design choices here: as for the image processing section, I tried to follow what has been done in the Demo, as this is the first time I am working with image data. As for the text processing layer, I tried to put to work the knowledge that I have gained in the Computational Semantics course, where we learned how to use LSTM models to approximate sentence meaning, and I thought that would be a good way to represent the captions, alongside BERT embeddings which, hopefully, give some more nuance to the word meanings for the LSTM to work on (also because we would not have enough data to train out own word embeddings on, perhaps). The classification layer is also similar to the one in Demo, albeit longer, and intended to gradually decrease the vector size down to one number representing the probability; the input to this layer is the concatenanted outputs of the two previous sections. The forward section specified how the data from the batch is processed; this one requires the device on which the model is to be specified, this is so that when a model is loaded and the device it is on is changed, it can still work (so you can train the model on a GPU but test it on a CPU or a different GPU). 
+ train() is a function that contains the training loop. It takes a list of loaders obtained in the collecting the training data section and trains the model over as many epochs as there are loaders (due to how interconnected the data processing functions are, it was really difficult for me to include the dataloader generation *inside* the loop, but I would much rather have done it that way, as it also saves memory to have only one loader at a time, not 10 or 20). The loss function used here is binary cross-entropy loss as this is a binary classification problem; Adam is the selected optimizer. Loss is printed out at the end of every epoch, and every 25 batches a message is printed out to signify that training is still happening. This function returns a trained model. As for the learning rate here, initially it was higher and the model tended to return the same thing for all the instances in a batch. Only lowering it made it possible to get reasonable output.
+ save_model() and load_model() use the pickle module to save and load trained models.
+ test_model() takes a trained model and a test loader list (with only one loader), as well as the desired device; it then sets the model in eval mode and runs it over the testing data; predictions are encoded to reflect what class was predicted, and both the predictions and true classes are gradually represented as lists. This function returns those lists.
+ measures() takes the lists from test_model() and calculates different evaluation measures, like accuracy, precision, recall, f1.

### PART 3 - TRAINING AND TESTING  
Here multiple different combinations of data and hyperparameters were used (different batch sizes and epoch numbers, different data split sizes) to finally submit the best model, which is saved as model2k.pickle.

### PART 4 - EVALUATION AND ERROR ANALYSIS  
The best performance that I got was while running the 2k "smart" dataset/dataloader for 20 epochs. 1k, both smart and not (10 epochs) was not far behind, while the not "smart" 2k one was significantly worse. These were all run with the hyperparameters that are "built in" into the functions and classes in the modelA2.py file.  
The output of the measures() function for the testing of this best model was the following (it can also be seen in the notebook):  
`The following measures have been recorded for this model:
	Accuracy = 0.5433333333333333
	Recall = 0.8115942028985508
	Precision = 0.5022421524663677
	F1 = 0.6204986149584487`   
Although the test set is not precisely made up of 50% class 1 and 50% class zero (due to the way data was shuffled at an earlier point), this is still relatively decent performance. Looking at the recall and precision we can notice that while many (81%) of the matching caption-image pairs were retrieved correctly, many of the mismatched ones were also retrieved as matching (50.2% of all retrieved as positive were actually true positives). Clearly, the model is not that good at recognizing the negative class still (although without smart samples it was even worse).  
Four separate Pandas DataFrames were created and printed out to try to see if there is any trend in what the captions have in common that made them be classified or misclassified.  
From what I noticed, many of the properly classified image/caption pairs feature people and cooking or food (especially pizza), as well as kites and vehicles. The false positives feature many of the same items, but a number of them also is described as having computers or laptops as well as cats (or other animals) in the images. The true negatives also feature many of those categories, but in terms of vocabulary seem to feature rarer words ('perched', 'squash', 'skier', 'marble', 'yams', 'crammed', 'skillet'). The same is the case for the false negatives ('art deco', 'grazing', 'appliances', 'cot') which may be what is perceived as a mismatch.  
Given that there isn't much correlation between what items system could recognize in the text and on images, experimenting with different image and text analysis elements could lead to an improvement. Perhaps the images were scaled down too much, and perhaps using a whole sentence representation as the final hidden states from an LSTM muddied up the entities that the system was supposed to recognize. Naturally, it would also be good to be able to run the model with more data, either to give it a better chance at learning something, or to have it highlight what the issues are even more. Similarly to experimenting with other image and text analysis, different proportions or methods of combining their outputs for classification could play a role too. Perhaps using word2vec vectors would also have had a better effect than using the BERT ones. What can be noticed from the DataFrames is also that the testing data was not split equally between the classes, and the same holds true for the training data. It would be better to have a more even split between the classes. From a practical standpoint, I would have prefered to have been able to generate dataloaders as a part of the training loop, but I have made it too complicated for myself.  

### BONUS PART - NEGATIVE SAMPLING  
Provided that I understood the clarification posted on Canvas well, I decided to implement this at the level of generating samples. Since the coco dataset only includes "correct" captions, the most "stupid" way of generating the incorrect class was shuffling the captions and images around, and making sure they did not end up back at the right picture (as there are 5 pairs that have the same picture). However, while a caption might not initially come from the same image, it may still be misleadingly similar, and, perhaps, might still be actually a decent caption. To avoid these "tricky" examples from making their way into the data (and thus ruining the system's definition of good vs. bad caption), I implemented an optional argument "smart" in create_samples() which then also looks at the coco classes that the images/captions belong to - so the salient entities featured in them, such as zebra, carrot, cat, or horse. The negative samples are then generated in such a way that the image and the mismatched caption do not share any classes. This means that there shouldn't be anything in the caption that overlaps in "meaning" with the image in terms of entities, hopefully making it easier for the system to match names to things in the images, so to say.  
When tested with 1k datasets, the difference between smart and not smart sampling was negligible. However, for the larger sets, it became more visible, and without the smart sampling the models performed really badly. 