import os
import json
import matplotlib.pyplot as plt


def show_obj(label_dir):
    files = os.listdir(label_dir)
    files.sort()

    attr_counter = {'weather': {}, 'scene': {}, 'timeofday': {}}
    obj_counter = {}
    light_counter = {}
    for file in files:
        with open(label_dir + '/' + file, 'r', encoding='utf-8') as f:
            lines = json.load(f)

            print('reading {}, {}'.format(file, len(lines)))

            for line in lines:
                weather = line['attributes']['weather']
                scene = line['attributes']['scene']
                timeofday = line['attributes']['timeofday']

                # 属性统计
                if weather in attr_counter['weather']:
                    attr_counter['weather'][weather] += 1
                else:
                    attr_counter['weather'][weather] = 1

                if scene in attr_counter['scene']:
                    attr_counter['scene'][scene] += 1
                else:
                    attr_counter['scene'][scene] = 1

                if timeofday in attr_counter['timeofday']:
                    attr_counter['timeofday'][timeofday] += 1
                else:
                    attr_counter['timeofday'][timeofday] = 1

                labels = line['labels']
                for label in labels:
                    category = label['category']

                    if category == 'lane' or category == 'drivable area':
                        continue

                    # 类别统计
                    if category in obj_counter:
                        obj_counter[category] += 1
                    else:
                        obj_counter[category] = 1

                    # 红绿灯统计
                    if category == 'traffic light':
                        light = label['attributes']['trafficLightColor']
                        if light in light_counter:
                            light_counter[light] += 1
                        else:
                            light_counter[light] = 1

    print('attributes counter: ', attr_counter)
    print('objects counter: ', obj_counter)
    print('lights counter: ', light_counter)

    plt.figure(figsize=(15, 9))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.025, right=0.99, hspace=0.25, wspace=0.1)

    plt.subplot(3, 1, 1)
    plt.title('obj', fontsize=12)
    for k, v in obj_counter.items():
        bar = plt.bar(k, v, log=True)
        plt.bar_label(bar, fontsize=6)

    plt.subplot(3, 2, 3)
    plt.title('light', fontsize=12)
    for k, v in light_counter.items():
        bar = plt.bar(k, v, log=True)
        plt.bar_label(bar, fontsize=6)

    plt.subplot(3, 2, 4)
    plt.title('timeofday', fontsize=12)
    for k, v in attr_counter['timeofday'].items():
        bar = plt.bar(k, v, log=True)
        plt.bar_label(bar, fontsize=6)

    plt.subplot(3, 2, 5)
    plt.title('weather', fontsize=12)
    for k, v in attr_counter['weather'].items():
        bar = plt.bar(k, v, log=True)
        plt.bar_label(bar, fontsize=6)

    plt.subplot(3, 2, 6)
    plt.title('scene', fontsize=12)
    for k, v in attr_counter['scene'].items():
        bar = plt.bar(k, v, log=True)
        plt.bar_label(bar, fontsize=6)

    plt.show()


def show_lane_drivable(label_dir):
    files = os.listdir(label_dir)
    files.sort()

    lane_counter = {'direction': {}, 'style': {}, 'type': {}}
    drivable_counter = {}
    for file in files:
        with open(label_dir + '/' + file, 'r', encoding='utf-8') as f:
            lines = json.load(f)

            print('reading {}, {}'.format(file, len(lines)))

            for line in lines:
                labels = line['labels']

                for label in labels:
                    category = label['category']

                    # 可行驶区域统计
                    if category == 'drivable area':
                        type = label['attributes']['areaType']
                        if type in drivable_counter:
                            drivable_counter[type] += 1
                        else:
                            drivable_counter[type] = 1

                    # 车道线统计
                    if category == 'lane':
                        direction = label['attributes']['laneDirection']
                        style = label['attributes']['laneStyle']
                        type = label['attributes']['laneType']

                        if direction in lane_counter['direction']:
                            lane_counter['direction'][direction] += 1
                        else:
                            lane_counter['direction'][direction] = 1

                        if style in lane_counter['style']:
                            lane_counter['style'][style] += 1
                        else:
                            lane_counter['style'][style] = 1

                        if type in lane_counter['type']:
                            lane_counter['type'][type] += 1
                        else:
                            lane_counter['type'][type] = 1

    print('drivable_counter: ', drivable_counter)
    print('lane_counter: ', lane_counter)

    plt.figure(figsize=(15, 9))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.025, right=0.99, hspace=0.25, wspace=0.4)

    plt.subplot(2, 3, 1)
    plt.title('drivable type', fontsize=12)
    for k, v in drivable_counter.items():
        bar = plt.bar(k, v, log=True)
        plt.bar_label(bar, fontsize=6)

    plt.subplot(2, 3, 2)
    plt.title('lane direction', fontsize=12)
    for k, v in lane_counter['direction'].items():
        bar = plt.bar(k, v, log=True)
        plt.bar_label(bar, fontsize=6)

    plt.subplot(2, 3, 3)
    plt.title('lane style', fontsize=12)
    for k, v in lane_counter['style'].items():
        bar = plt.bar(k, v, log=True)
        plt.bar_label(bar, fontsize=6)

    plt.subplot(2, 1, 2)
    plt.title('lane type', fontsize=12)
    for k, v in lane_counter['type'].items():
        bar = plt.bar(k, v, log=True)
        plt.bar_label(bar, fontsize=6)

    plt.show()


label_dir = '../bdd100k/labels'

show_obj(label_dir)
show_lane_drivable(label_dir)