#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tensorflow as tf
import tensorflow.keras as keras

class NoiseRemover():
    def _erase_circles(img, circles):
        circles = circles[0] # hough circles returns a nested list for some reason
        for circle in circles:
            x = circle[0] # x coordinate of circle's center
            y = circle[1] # y coordinate of circle's center
            r = circle[2] # radius of circle

            img = cv2.circle(img, (int(x), int(y)), radius = int(r), color = (255,0,0), thickness = 2) # erase circle by making it white (to match the image background)
        return img


    def _detect_and_remove_circles(img):
        hough_circle_locations = cv2.HoughCircles(img, method = cv2.HOUGH_GRADIENT, dp = 1, minDist = 1, param1 = 50, param2 = 5, minRadius = 0, maxRadius = 2)
        if hough_circle_locations is not None:
            img = NoiseRemover._erase_circles(img, hough_circle_locations)
            pass
        return img

    def remove_all_noise(img):
        # run some basic tests to get rid of easy-to-remove noise -- first pass
        img = img[1]
        img = ~img # white letters, black background
        img = cv2.erode(img, numpy.ones((3, 2), numpy.uint8), iterations = 1) # weaken circle noise and line noise
        img = ~img # black letters, white background
        # img = scipy.ndimage.median_filter(img, (5, 1)) # remove line noise
        # img = scipy.ndimage.median_filter(img, (1, 3)) # weaken circle noise
        img = cv2.erode(img, numpy.ones((2, 2), numpy.uint8), iterations = 1) # dilate image to initial stage (erode works similar to dilate because we thresholded the image the opposite way)
        # img = scipy.ndimage.median_filter(img, (3, 3)) # remove any final 'weak' noise that might be present (line or circle)

        # detect any remaining circle noise
        # img = NoiseRemover._detect_and_remove_circles(img) # after dilation, if concrete circles exist, use hough transform to remove them
        # eradicate any final noise that wasn't removed previously -- second pass
        img = cv2.dilate(img, numpy.ones((3, 3), numpy.uint8), iterations = 1) # actually performs erosion
        # img = scipy.ndimage.median_filter(img, (3, 1)) # finally completely remove any extra noise that remains
        img = cv2.dilate(img, numpy.ones((2, 2), numpy.uint8), iterations = 1) # erode just a bit to polish fine details
        img = cv2.erode(img, numpy.ones((2, 2), numpy.uint8), iterations = 2) # dilate image to make it look like the original
        return img


def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    outputs = ()

    with tf.device('/cpu:0'):
        with open(args.output, 'w') as output_file:
            json_file = open(args.model_name + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.load_weights(args.model_name+'.h5')
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.legacy.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])

            for x in os.listdir(args.captcha_dir):
                # load image and preprocess it
                raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
                if raw_data is not None :
                    # Add some preprocessing
                    # denoise
                    raw_data1 = cv2.threshold(raw_data, 222, 255, cv2.THRESH_BINARY)
                    raw_data2 = NoiseRemover.remove_all_noise(raw_data1)
                    rgb_data = cv2.cvtColor(raw_data2, cv2.COLOR_BGR2RGB)
                    image = numpy.array(rgb_data) / 255.0
                    (c, h, w) = image.shape
                    image = image.reshape([-1, c, h, w])
                    prediction = model.predict(image)
                    # write to file here
                    output_file.write(x + "," + decode(captcha_symbols, prediction) + "\n")
                    # output_file.write(decode(captcha_symbols, prediction) + ",")
                    # outputs.append(x + ",", decode(captcha_symbols, prediction))
                    print('Classified ' + x)
                else:
                    print('File not found for - ' + args.captcha_dir + x)
        # sort outputs
        
        # write to file

    # in main




if __name__ == '__main__':
    main()
