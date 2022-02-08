import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def detect_person(image_input):
    import argparse
    import time

    from PIL import Image
    from PIL import ImageDraw

    from pycoral.adapters import common
    from pycoral.adapters import detect
    from pycoral.utils.dataset import read_label_file
    from pycoral.utils.edgetpu import make_interpreter
    label_path = os.path.join(BASE_DIR, 'coral_files', 'coco_labels.txt')
    model_path = os.path.join(BASE_DIR, 'coral_files', 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
    print(model_path)

    labels = read_label_file(label_path)
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    #image = Image.fromarray(image_input)
    image = image_input
    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    interpreter.invoke()
    objs = detect.get_objects(interpreter, 0.4, scale)
    for obj in objs:
        print(labels.get(obj.id, obj.id))
        print('  id:    ', obj.id)
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)


if __name__ == '__main__':
    from PIL import Image
    img = Image.open('coral_files/020050.jpg')
    detect_person(img)

