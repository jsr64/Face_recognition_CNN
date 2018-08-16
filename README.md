# Face_recognition_CNN

Earlier i tried face recognition using haar cascade....it just used a haar classifier to find faces in an image..

In this project i would attempt face recognition with convoluted neural nets .

### Steps i went through:
    First go through [this]:https://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/
    https://www.youtube.com/watch?v=4eIBisqx9_g
    
    to detect an object we use some technique to find objects inside an image like:
      1).Sliding window --> we slide a window over an image and use cnn to find object in the window
      2).RCNN
      3).SSD
      4).YOLO
### I would be using YOLO just because it sounds good and i have no hands on experience with No 2,3,4
#### sould not make any difference since i would need to start from scratch

--------------------
sooooo i watched some videos about these algorithms found a good repo named 'darkflow' cloning it and would be implementing it.
REPO : https://github.com/thtrieu/darkflow

so now i have got a good idea about darkflow.
### my idea:
        i would be using transfer lerning kind of thing .... where i would use darkflows YOLO to get a json of persons.
        Then i would be using these coordinates to my own convoluted neural net to recognise images of only me.
        and cateogrise other people as OTHERS.
        advantage it that YOLO would most probably provide me images of people then further i could classify people.
        Like me or some one other.
        
Right now i am stuck at generating my own dataset of my image...have collected 100 images of myself then i would be appliing some transformations on those images to generate more image so as to increase my data set.
