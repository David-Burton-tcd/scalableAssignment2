# scalableAssignment2

To Run;
Ensure the following are installed using pip
- opencv-python
- numpy
- scipy
- tensorflow
- Captcha

Once these have been installed, you should be able to run the above files. Please see below some example commands for how to run them;

python3 generate.py --width 128 --height 64 --max-length 6 --symbols symbols.txt --count 50000 --output-dir test
- the above runs the generate script - this generates 50,000 catpcha symbols using the jjester font (hard coded into the file) and outputs the images to a directory named test. The only symbols used to generate these images are in a file called symbols.txt

- python3 train2.py --width 128 --height 64 --length 6 --symbols symbols.txt --batch-size 64 --epochs 50 --output-model test3 --train-dataset train --validate-dataset validate
The above runs the train2 script. This will train a CNN model using training images from train directory with validation images from validation directory. It outputs a model named test3 as a keras model (.h5 file extenson).
It takes a batch size of 64 images and it trains for 50 epochs (the code contains instructions to terminate early should certain patience parameters be met).

- python3 classify.py --model-name test5 --captcha-dir dataset --output results/run1.csv --symbols symbols.txt
This runs the classify script which takes a trained model and compares it versus a directory of images. It outputs a csv to the results directory in a file called run1.csv.
