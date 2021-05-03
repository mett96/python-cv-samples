"""
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
"""
import json
import os
import traceback

from cv2 import cv2
import numpy as np
# import paho.mqtt.client as mqtt
import time


def avg_circles(circles, b):
    avg_x = 0
    avg_y = 0
    avg_r = 0
    for i in range(b):
        # optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x / b)
    avg_y = int(avg_y / b)
    avg_r = int(avg_r / b)
    return avg_x, avg_y, avg_r


def dist_2_pts(x1, y1, x2, y2):
    # print np.sqrt((x2-x1)^2+(y2-y1)^2)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calibrate_gauge(gauge_number, file_type):
    """
    This function should be run using a test image in order to calibrate the range available to the dial as well as
    the units.  It works by first finding the center point and radius of the gauge.  Then it draws lines at hard
    coded intervals (separation) in degrees.  It then prompts the user to enter position in degrees of the lowest
    possible value of the gauge, as well as the starting value (which is probably zero in most cases but it won't
    assume that).  It will then ask for the position in degrees of the largest possible value of the gauge. Finally,
    it will ask for the units.  This assumes that the gauge is linear (as most probably are). It will return the min
    value with angle in degrees (as a tuple), the max value with angle in degrees (as a tuple), and the units (as a
    string).
    """

    img = get_img(gauge_number, file_type)
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.medianBlur(gray, 5)
    cv2.imshow('image', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # for testing, output gray image
    # cv2.imwrite('gauge-%s-bw.%s' %(gauge_number, file_type),gray)
    save_img(gray, 'gauge-%s-bw.%s' % (gauge_number, file_type))

    img_canny = cv2.Canny(gray, 100, 200)
    save_img(img_canny, 'gauge-%s-canny.%s' % (gauge_number, file_type))

    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    # Draw contours
    for i in range(len(contours)):
        cv2.drawContours(img, contours, i, (0, 255, 0), 2, cv2.LINE_8, hierarchy, 0)

    # detect circles
    # restricting the search from 35-48% of the possible radii gives fairly good results across different samples.
    # Remember that these are pixel values which correspond to the possible radii search range.
    img_canny = np.uint8(img_canny)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=50,
                               minRadius=int(height * 0.35), maxRadius=int(height * 0.48))

    # for debug print all circles
    print(circles.shape)
    img = get_img(gauge_number, file_type)
    for (x, y, r) in circles[0]:
        r = int(r)
        cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)
    save_img(img, 'gauge-%s-circles-test.%s' % (gauge_number, file_type))

    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    a, b, c = circles.shape
    print("Circle shape:", a, b, c)
    x, y, r = avg_circles(circles, b)
    print("Average circle", x, y, r)

    img = get_img(gauge_number, file_type)
    # draw center and circle
    cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle

    # for testing, output circles on image
    # cv2.imwrite('gauge-%s-circles.%s' % (gauge_number, file_type), img)
    save_img(img, 'gauge-%s-circles.%s' % (gauge_number, file_type))

    # for calibration, plot lines from center going out at every 10 degrees and add marker
    # for i from 0 to 36 (every 10 deg)

    '''
    goes through the motion of a circle and sets x and y values based on the set separation spacing.  Also adds 
    text to each line.  These lines and text labels serve as the reference point for the user to enter NOTE: by 
    default this approach sets 0/360 to be the +x axis (if the image has a cartesian grid in the middle), 
    the addition (i+9) in the text offset rotates the labels by 90 degrees so 0/360 is at the bottom (-y in 
    cartesian).  So this assumes the gauge is aligned in the image, but it can be adjusted by changing the value of 9 
    to something else. 
    '''
    separation = 10.0  # in degrees
    interval = int(360 / separation)
    p1 = np.zeros((interval, 2))  # set empty arrays
    p2 = np.zeros((interval, 2))
    p_text = np.zeros((interval, 2))
    for i in range(0, interval):
        for j in range(0, 2):
            if j % 2 == 0:
                p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180)  # point for lines
            else:
                p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
    text_offset_x = 10
    text_offset_y = 5
    for i in range(0, interval):
        for j in range(0, 2):
            if j % 2 == 0:
                p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos(
                    separation * (i + 9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees
            else:
                p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                p_text[i][j] = y + text_offset_y + 1.2 * r * np.sin(
                    separation * (i + 9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees

    # add the lines and labels to the image
    for i in range(0, interval):
        cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])), (0, 255, 0), 2)
        cv2.putText(img, '%s' % (int(i * separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (0, 0, 0), 1, cv2.LINE_AA)

    # cv2.imwrite('gauge-%s-calibration.%s' % (gauge_number, file_type), img)
    save_img(img, 'gauge-%s-calibration.%s' % (gauge_number, file_type))

    # get user input on min, max, values, and units
    print('gauge number: %s' % gauge_number)

    setts = json.load(open("sets.json"))
    key = str(gauge_number)
    if key in setts:
        print("loading settings...")
        min_angle = setts[key]['min_angle']
        max_angle = setts[key]['max_angle']
        min_value = setts[key]['min_value']
        max_value = setts[key]['max_value']
        units = setts[key]['units']
    else:
        min_angle = input('Min angle (lowest possible angle of dial) - in degrees: ')  # the lowest possible angle
        max_angle = input('Max angle (highest possible angle) - in degrees: ')  # highest possible angle
        min_value = input('Min value: ')  # usually zero
        max_value = input('Max value: ')  # maximum reading of the gauge
        units = input('Enter units: ')

        setts[gauge_number] = {
            'min_angle': min_angle,
            'max_angle': max_angle,
            'min_value': min_value,
            'max_value': max_value,
            'units': units
        }
        json.dump(setts, open("sets.json", 'w'), indent=2)

    # for testing purposes: hardcode and comment out inputs above
    # min_angle = 45
    # max_angle = 320
    # min_value = 0
    # max_value = 200
    # units = "PSI"

    return min_angle, max_angle, min_value, max_value, units, x, y, r


def get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, gauge_number, file_type):
    # for testing purposes
    # img = cv2.imread('gauge-%s.%s' % (gauge_number, file_type))

    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set threshold and maxValue
    thresh = 175
    maxValue = 255

    # for testing purposes, found cv2.THRESH_BINARY_INV to perform the best
    # th, dst1 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY);
    # th, dst2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV);
    # th, dst3 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_TRUNC);
    # th, dst4 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_TOZERO);
    # th, dst5 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_TOZERO_INV);
    # cv2.imwrite('gauge-%s-dst1.%s' % (gauge_number, file_type), dst1)
    # cv2.imwrite('gauge-%s-dst2.%s' % (gauge_number, file_type), dst2)
    # cv2.imwrite('gauge-%s-dst3.%s' % (gauge_number, file_type), dst3)
    # cv2.imwrite('gauge-%s-dst4.%s' % (gauge_number, file_type), dst4)
    # cv2.imwrite('gauge-%s-dst5.%s' % (gauge_number, file_type), dst5)

    # apply thresholding which helps for finding lines
    th, dst2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV)

    # found Hough Lines generally performs better without Canny / blurring, though there were a couple exceptions
    # where it would only work with Canny / blurring
    # dst2 = cv2.medianBlur(dst2, 5)
    # dst2 = cv2.Canny(dst2, 50, 150)
    # dst2 = cv2.GaussianBlur(dst2, (5, 5), 0)

    # for testing, show image after thresholding
    # cv2.imwrite('gauge-%s-tempdst2.%s' % (gauge_number, file_type), dst2)
    save_img(dst2, 'gauge-%s-tempdst2.%s' % (gauge_number, file_type))

    # find lines
    minLineLength = 10
    maxLineGap = 0
    lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180, threshold=100, minLineLength=minLineLength,
                            maxLineGap=0)  # rho is set to 3 to detect more lines, easier to get more then filter them out later

    # for testing purposes, show all found lines
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.imwrite('gauge-%s-lines-test.%s' %(gauge_number, file_type), img)
    save_img(img, 'gauge-%s-lines-test.%s' % (gauge_number, file_type))

    # remove all lines outside a given radius
    final_line_list = []
    # print "radius: %s" %r

    diff1LowerBound = 0.0  # diff1LowerBound and diff1UpperBound determine how close the line should be from the center
    diff1UpperBound = 0.4
    diff2LowerBound = 0.5  # diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
    diff2UpperBound = 1.0
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
            diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle
            # set diff1 to be the smaller (closest to the center) of the two), makes the math easier
            if diff1 > diff2:
                diff1, diff2 = diff2, diff1
            # check if line is within an acceptable range
            if (((diff1 < diff1UpperBound * r) and (diff1 > diff1LowerBound * r) and (
                    diff2 < diff2UpperBound * r)) and (diff2 > diff2LowerBound * r)):
                line_length = dist_2_pts(x1, y1, x2, y2)
                # add to final list
                final_line_list.append([x1, y1, x2, y2])

    # testing only, show all lines after filtering
    img = get_img(gauge_number, file_type)
    for i in range(len(final_line_list)):
        x1 = final_line_list[i][0]
        y1 = final_line_list[i][1]
        x2 = final_line_list[i][2]
        y2 = final_line_list[i][3]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    save_img(img, 'gauge-%s-lines-test-2.%s' % (gauge_number, file_type))

    if len(final_line_list) == 0:
        print("NO GOOD LINE FOUND")
        return "?"

    img = get_img(gauge_number, file_type)
    # assumes the first line is the best one
    x1 = final_line_list[0][0]
    y1 = final_line_list[0][1]
    x2 = final_line_list[0][2]
    y2 = final_line_list[0][3]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # for testing purposes, show the line overlayed on the original image
    save_img(img, 'gauge-%s-line.%s' % (gauge_number, file_type))

    # find the farthest point from the center to be what is used to determine the angle
    dist_pt_0 = dist_2_pts(x, y, x1, y1)
    dist_pt_1 = dist_2_pts(x, y, x2, y2)
    if dist_pt_0 > dist_pt_1:
        x_angle = x1 - x
        y_angle = y - y1
    else:
        x_angle = x2 - x
        y_angle = y - y2
    # take the arc tan of y/x to find the angle
    res = np.arctan(np.divide(float(y_angle), float(x_angle)))
    # np.rad2deg(res) #coverts to degrees

    # print x_angle
    # print y_angle
    # print res
    # print np.rad2deg(res)

    # these were determined by trial and error
    res = np.rad2deg(res)
    if x_angle > 0 and y_angle > 0:  # in quadrant I
        final_angle = 270 - res
    if x_angle < 0 and y_angle > 0:  # in quadrant II
        final_angle = 90 - res
    if x_angle < 0 and y_angle < 0:  # in quadrant III
        final_angle = 90 - res
    if x_angle > 0 and y_angle < 0:  # in quadrant IV
        final_angle = 270 - res

    # print final_angle

    old_min = float(min_angle)
    old_max = float(max_angle)

    new_min = float(min_value)
    new_max = float(max_value)

    old_value = final_angle

    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value


def get_img(gauge_number, file_type):
    # print(os.getcwd())
    img = cv2.imread(os.path.join("images", 'gauge-%s.%s' % (gauge_number, file_type)))

    return img


def save_img(image, file_name: str):
    folder = os.path.join("images", "executions")
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, file_name), image)


def main(gauge_number=1, file_type='jpg'):
    print("Gauge number: ", gauge_number)
    # name the calibration image of your gauge 'gauge-#.jpg', for example 'gauge-5.jpg'.  It's written this way so
    # you can easily try multiple images
    min_angle, max_angle, min_value, max_value, units, x, y, r = calibrate_gauge(gauge_number, file_type)

    # feed an image (or frame) to get the current value, based on the calibration, by default uses same image as
    # calibration
    img = get_img(gauge_number, file_type)
    val = get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, gauge_number, file_type)
    print("Current reading: %s %s" % (val, units))


def main_all():
    max_item = 11

    for gauge_number in range(1, max_item + 1):
        try:
            main(gauge_number)
        except:
            traceback.print_exc()


if __name__ == '__main__':
    gauge_number = 11
    main(gauge_number)
    # main_all()
