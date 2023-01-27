import json
import os
import cv2


def show_obj(label_dir, image_dir):
    cls_colors = {
        'car': [0, 200, 255],  # orange
        'motor': [0, 200, 255],  # orange
        'bus': [0, 200, 255],  # orange
        'truck': [0, 200, 255],  # orange
        'train': [0, 200, 255],  # orange
        'traffic sign': [0, 255, 0],  # green
        'traffic light': [0, 255, 0],  # green
        'rider': [0, 0, 255],  # red
        'person': [0, 0, 255],  # red
        'bike': [0, 0, 255],  # red
    }

    files = os.listdir(label_dir)
    files.sort()

    for file in files:
        with open(label_dir + '/' + file, 'r', encoding='utf-8') as f:
            lines = json.load(f)

            print('reading {}, {}'.format(file, len(lines)))

            sub_dir = file.split('_')[-1].replace('.json', '')
            for line in lines:
                name = line['name']
                labels = line['labels']
                image = cv2.imread(image_dir + '/' + sub_dir + '/' + name)

                for label in labels:
                    category = label['category']
                    if category in cls_colors:
                        box2d = label['box2d']
                        pt1 = (round(box2d['x1']), round(box2d['y1']))
                        pt2 = (round(box2d['x2']), round(box2d['y2']))
                        pt3 = (round(box2d['x1']), round(box2d['y1'] - 2))

                        text = category if category != 'traffic light' else label['attributes']['trafficLightColor']

                        cv2.rectangle(image, pt1, pt2, cls_colors[category])
                        cv2.putText(image, text, pt3, 2, 0.5, cls_colors[category])

                print('image: ', name)

                cv2.namedWindow(name, 0)
                cv2.resizeWindow(name, 2560, 1440)
                cv2.imshow(name, image)
                cv2.waitKey(10 * 60 * 1000)
                cv2.destroyAllWindows()


label_dir = '../bdd100k/labels'
image_dir = '../bdd100k/images'

show_obj(label_dir, image_dir)
