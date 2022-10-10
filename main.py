import cv2
from os import listdir, system
from os.path import isfile, join

SCALE_FACTOR = 0.2
 
#define the model optimization variables
STARTING_NOSE_X = 625
STARTING_NOSE_Y = 1000
MAX_NOSE_W = 875
MAX_NOSE_H = 750

# Load the models
nose_model = cv2.CascadeClassifier('./models/nose.xml')

folder = "./images"
sample_images = [f for f in listdir(folder) if isfile(join(folder, f))]

#get base resolution
example_image = cv2.imread(f"{folder}/{sample_images[0]}")
BASE_RESOLUTION = (example_image.shape[1], example_image.shape[0])

def run_detection(image, value=None):
    #crop the image
    cropped = image[STARTING_NOSE_Y:STARTING_NOSE_Y+MAX_NOSE_H, STARTING_NOSE_X:STARTING_NOSE_X+MAX_NOSE_W]

    #change to gray tons
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    #run detection
    nose_rects = nose_model.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(300, 300)
    )

    #if we do not find any hits return an empty tuple
    if not len(nose_rects):
        return ()

    #sort them based on the x coordinate so that we can later choose the one 
    #that is in the middle
    sorted_noses = sorted(nose_rects, key=lambda element: element[0])

    #choose the middle one
    middle_index = (len(sorted_noses) // 2)
     
    #get coords for (x,y,w,h) the middle nose:
    (x,y,w,h) = sorted_noses[middle_index-1]
    
    #normalize coords to the original size before being cropped
    return (x+STARTING_NOSE_X, y+STARTING_NOSE_Y, w, h)


def run_ratio_test():
    yes = 0
    no = 0

    for i in sample_images:
        image = cv2.imread(f"{folder}/{i}")

        nose_coords = run_detection(image)

        if len(nose_coords) :
            yes += 1
        else:
            no += 1

        current_ratio = yes / (yes + no)
        
        system(f"title [{sample_images.index(i)}] - (cr:{round(current_ratio, 2)})")

    return current_ratio

def show_images():

    time_line = []
    i = 0

    while i < len(sample_images):
        image_name = sample_images[i]

        image = cv2.imread(f"{folder}/{image_name}")
        nose_coords = run_detection(image)

        if not len(nose_coords):
            i += 1
            continue

        #append the current image to the "time" line
        elif i not in time_line:
            time_line.append(i)

        #draw the circle on the nose
        cv2.circle(image, (int(nose_coords[0]+(nose_coords[2]/2)), int(nose_coords[1]+(nose_coords[3]/2))), 15, (0,0,255), -1)

        #scale down to display
        image = cv2.resize(image, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA)

        #display the image with the circle
        cv2.imshow(f"{nose_coords}", image)

        #controls
        while True:
            pressed_key = cv2.waitKey(0)

            if pressed_key == 8: #backspace - exit
                quit()

            elif pressed_key == 113: #q - previous
                i = time_line[time_line.index(i) - 1]
                break

            elif pressed_key == 101: #e - next
                i += 1
                break

        cv2.destroyAllWindows()

def brute_force_test():
    #brute scales
    best_value = 0
    best_ratio = 0
 
    for value in range(50, 150, 10):
        current_ratio = run_ratio_test(value)

        #check if it is better
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_value = value

        print(f"Best value: {best_value} with ratio: {best_ratio}")

    input()

def get_compile_resolution():
    max_x_deviation = 0
    max_y_deviation = 0

    for i in sample_images:
        image = cv2.imread(f"{folder}/{i}")

        nose_coords = run_detection(image)

        if not (len(nose_coords)):
            continue

        (x, y) = nose_coords[0], nose_coords[1]

        x_deviation = abs(x - BASE_RESOLUTION[0]/2)
        y_deviation = abs(y - BASE_RESOLUTION[1]/2)

        if x_deviation > max_x_deviation:
            max_x_deviation = x_deviation

        if y_deviation > max_y_deviation:
            max_y_deviation = y_deviation

        print(f"{sample_images.index(i)}")

    return (BASE_RESOLUTION[0] - max_x_deviation, BASE_RESOLUTION[1] - max_y_deviation)


if __name__ == "__main__":
    show_images()