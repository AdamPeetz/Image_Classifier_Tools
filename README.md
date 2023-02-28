 # Custom Image Classifier Tools 

The number of implementations for image classifiers is limited only by the imagination of the people who can create them. They can be used for tasks as mundane as labeling key moments in enormous amounts of video, or more abstract things like predicting wait time from pictures of lines at a ski lift. These varied ideas cannot be implemented without the creation of many individual pieces of code to support them. Scraping programs for the collection of image data. Cleaning code to improve that data's quality. High performance models to perform the classification. All of these come together to build an image classification model. Creating a generalized toolkit containing all these pieces of code can support the creation of image classifiers and drive value for anyone attempting to leverage the power of machine learning and AI for a wide number of tasks.  

The goal of this project is to demonstrate a data science pipeline for custom image classification problems. In this example, an image classification deep neural network will be built to classify images of tanks and cars from the dataset up. 

![alt text](https://github.com/AdamPeetz/imagehosting/blob/main/Cars%20and%20Tanks%20.png) <br>

# Dataset Creation
An image classifier requires labeled images of objects to train and test a model. Robust classification models often train on millions of labeled images. For example, the ImageNet classification dataset has more than 14 million images belonging to 21 thousand different classes (Fei-Fei et al, 2023). 	Sourcing many labeled images can be a challenge for developing an image classification model. Social media is one place where many images can be found, pre labeled with hashtags, titles, or organized into dedicated topic forums. Reddit hosts several forums dedicated to images of specific objects, two examples of this are r/carporn and r/tankporn. These subreddits exclusively contain high quality images of tanks and cars and will be the source of data used in this model. 

Reddit provides users with access to scrape their website with an API. Using the reddit API to obtain image data ensures that it is collected ethically as it adheres to the website's data access policy. The results of the scraping operation are summarized in the tables below.  

Scraping notebook: https://github.com/AdamPeetz/CustomImageClassifierTools/blob/main/reddit_scraper.ipynb 

Cleaning notebook: https://github.com/AdamPeetz/CustomImageClassifierTools/blob/main/duplicate_remover.ipynb 

![alt text](https://github.com/AdamPeetz/imagehosting/blob/main/scraping%20and%20cleaning.jpg) <br>

# Data Preparation 

Images scraped from the internet come in many different shapes and sizes. Convolutional neural networks (CNNs) need images to be scaled to a specific size before they can be input into the machine learning model. Scaling down images also reduces the amount of storage space they consume. This allows the model to use fewer resources during its training phase. 
 
AlexNet, a seminal CNN research effort by Krizhevsky et al. (2017), employed an image processing pipeline that downscaled images to 256x256 prior to inputting them in the model.  In this pipeline, rectangular images were rescaled around the shortest side, and then the center was cropped out of them. The authors also state that images should have the RGB values of the images centered, which reduces bias large pixel values may introduce to the weights of the model. The centering step is not included in the preprocessing code and is instead performed by the Keras API prior to loading the images into the model. Centering the pixel values during this stage of preprocessing will result in a series of black squares. These black squares can be used to successfully train the model but do not create an image set that makes sense to humans. An example of the scaling operation is shown below. 

Resizing notebook: 

https://github.com/AdamPeetz/CustomImageClassifierTools/blob/main/resizing_images.ipynb 

The rescaling operation reduces the overall dataset size: 

![alt text](https://github.com/AdamPeetz/imagehosting/blob/main/date%20resizing%20effects.jpg) <br>

## Image Scaling: 
![alt text](https://github.com/AdamPeetz/imagehosting/blob/main/image%20resizing.jpg) <br>

The processed dataset used in this project is available on Kaggle: https://www.kaggle.com/datasets/gatewayadam/cars-and-tanks-image-classification. 

# Data Augmentation 

There are 1,303 images in the training dataset. This is a small dataset when compared to those used in competitions like ImageNet. Image classification models are known to benefit from large datasets. The number of images used for training can be expanded beyond the original 1,303 with data augmentation. 

Augmenting images creates small adjustments to each image. Examples are adjusting the brightness of an image or flipping it horizontally or vertically. The best way to illustrate why this works is with an example. If the network has only seen images of cars taken in bright light it may assume the brightness of an image is a defining feature of a car. Adjusting the brightness of images as part of the augmentation pipeline will expand the dataset and help the model identify images with varying brightness. 

The effects of the augmentation can be seen below. 

![alt text](https://github.com/AdamPeetz/imagehosting/blob/main/augmentation%20example.jpg) <br>

# Model Architectures 
Three model architectures were tested. A VGG style sequential network, a custom Peetz Net architecture built with the functional API, and an Xception transfer learning model.  

## VGG Sequential  
The VGG network design was proposed by Karen Simonyan and Andrew Zisserman in 2015. A VGG network is built around stacked convolutional blocks which are separated by a maxpooling layers. The number of convolutional filters doubles in each layer before being flattened and fed into a fully connected network that translates the patterns detected by the convolutional layers into a prediction. The VGG network architecture is designed to show the effects increasing network depth has on prediction accuracy. Its straightforward design works well as a baseline model for evaluating the performance of more complex designs. 

VGG Training Notebook: https://github.com/AdamPeetz/CustomImageClassifierTools/blob/main/cnn_cars_tanks_VGG.ipynb 

## Peetz Net Functional  
Peetz Net was designed by the author for high accuracy classification of the CIFAR dataset. It was designed to generate two separate feature extractor stacks that identify radically different patterns in an image. One stack starts with large filters that are reduced in size as the network feeds down, the other starts with small filters that are increased in size towards the bottom of the network. The conclusions of those two extractor stacks are concatenated and flattened before being fed into fully connected layers for the classification of cars or tanks.  

Peetz Net Training Notebook: https://github.com/AdamPeetz/CustomImageClassifierTools/blob/main/cnn_cars_tanks_peetznet.ipynb 

## Xception Transfer Learning 
Transfer learning is one of the most powerful techniques in deep learning. Transfer learning takes the pretrained weights of an existing model and applies them to a new classification problem. It lets network designers use the resources and pattern recognition of professional models such as Xception for their own unique tasks. The Xception network is known for its extensive use of depth wise convolutions. Xception detects cross channel correlations separate from special correlations to achieve outstanding accuracy on the ImageNet classification test. (Chollet, 2016) 

Xception Training Notebook: https://github.com/AdamPeetz/CustomImageClassifierTools/blob/main/cnn_cars_tanks_xception.ipynb 

Dropout has been added as a regularization technique to combat early convergence in all model designs. 

## Dropout Regularization
Dropout randomly inactivates a percentage of the connections in a layer during each step of training. This prevents the model from over developing any one pathway in the neural network. Dropout effectively creates an ensemble of weak predictors inside the model which helps generalize it for never-before-seen data. Excessive dropout will make the model fail to converge on a solution. When applied in the right dose, it will prevent early convergence in the model. 

# Model Performance 
The Xception learning model was the highest performing model of the set, achieving 98% accuracy on the test dataset. The Xception model weights are saved and exported for use in future prediction of cars and tanks. A summary of all three models’ performance is shown in the table below. Additional performance discussion is available in the notebooks created for each model. 

![alt text](https://github.com/AdamPeetz/imagehosting/blob/main/model%20accuracy.jpg) <br>

# Ad Hoc Predictions
The tools used in this pipeline can be brought together to create ad hoc predictions of tanks and cars for any image. First raw images are resized and center cropped to match the input size of the Xception model, which requires a 224x224 input. Then the saved weights of the pretrained model are loaded and used to generate a prediction against the cropped image. Examples of a few ad hoc predictions are shown below.  

Ad Hoc Predictor Notebook: https://github.com/AdamPeetz/CustomImageClassifierTools/blob/main/adhoc_predictor_volvo.ipynb 

![alt text](https://github.com/AdamPeetz/imagehosting/blob/main/ad-hoc%20examples.jpg) <br>

# Conclusion 
The data science pipeline used in this notebook is designed to give anyone the power to create and solve image classification problems. It includes code to scrap and clean images, notebook examples of how to train CNN models, and culminates in ad hoc prediction code that would allow a trained model to be deployed to make predictions on brand new images. If you found this repo helpful, please take a moment to follow me or give it a star. 

# References 
Abadi, A., Agarwal, P. B., Brevdo, E., Chen, Z., Citro, C., Corrado, G., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I.,Harp, A., Irving, G., Isard, M., Jozefowicz, R., Jia Y., Kaiser, L., Kudlur, M., Levenberg, J., Mané, D., Schuster, M., Monga, R., Moore, S. Murray, D., Olah, C., Shlens, J., Steiner, B., Sutskever, I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Viégas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., & Zheng. X., (2015). TensorFlow: Largescale machine learning on heterogeneous systems. Software available from tensorflow.org 
 
Chollet, Francois. (2015). Keras. GitHub. https://github.com/fchollet/keras  

Chollet, Francois. (2016). Xception: Deep Learning with Depthwise Separable Convolutions. DOI arXiv:1610.02357 

deeplizard. (2020). Create A Confusion Matrix For Neural Network Predictions. retreived 02/17/2023 from https://deeplizard.com/learn/video/km7pxKy4UHU 

Fei-Fei, L., Deng, J., Russakovsky, O., Berg, A., Li, Kai., (2023). Imagenet. Retrieved 02/22/2023 from   https://www.image-net.org/about.php 

Harris, C.R., Millman, K.J., van der Walt, S.J. et al. (2020) Array programming with NumPy. Nature 585, 357–362. DOI: 10.1038/s41586-020-2649-2 

Helwan, Abdulkader. (2021).Running AI Fashion Classification on Real Data. retreived 02/17/2023 from https://www.codeproject.com/Articles/5297329/Running-AI-Fashion-Classification-on-Real-Data 

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering. https://zenodo.org/badge/DOI/10.5281/zenodo.592536.svg 

Kapoor, Karnika. (2021). Pneumonia Diagnosis: Convnet Model. kaggle. retreived 02/17/2023 from https://www.kaggle.com/code/karnikakapoor/pneumonia-diagnosis-convnet-model/notebook 

Krizhevsky, Alex. Sutskever, Ilya. & Hinton, Geoffrey. (2017). ImageNet classification with deep convolutional 	neural netowkrs. Communications of the ACM. 60(8). DOI: 10.1145/3065386 

OpenCV Team. (2023). ComputerVision2 (cv2). Opencv.org 

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., and Grisel, O. and Blondel, M. and Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12 

r/carporn. (2023). datasource. reddit.com. retreived 01/20/2023 

r/tankporn. (2023). datasource. reddit.com. retreived 01/20/2023 

r/tanks. (2023). datasource. reddit.com. retreived 01/20/2023 

Regis Jesuit University. (2022). MSDS660 Deep Learning Cirriculum. Regis University 

Simonyan, Karen & Zisserman, Andrew. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. published as a conference paper at ICLR 2015. retreived 02/17/2023 from https://arxiv.org/abs/1409.1556 

Umesh, P. (2012). Image Processing in Python. CSI Communications, 23. 

 

 
