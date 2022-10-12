import cv2
import os
import json
import numpy as np

exts = ['.png', '.jpg', '.jpeg', '.bmp']

def FindColorList(img):
    if not len(img.shape) == 3:
        print("must input len(image.shape) is 3, but inpnt len(image.shape) is ", len(img.shape))
        return -1
    color_list = img.reshape(-1, img.shape[-1])
    color_list = np.unique(color_list, axis=0, return_counts=False)
    print("\n\n")
    print("\nCOLOR_LIST : ", color_list)
    print("\n\n")
    
    return color_list
    

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error:Creating directory.' + directory)


def search(dirname):
    filelists = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext in exts: 
            filelists.append(full_filename)
    return filelists

def createHullPoint(img, findcolor=[0, 0, 0]):
    h, w = img.shape[0:2]
    mask = (img == findcolor).all(axis=2) # peak
    maskimg = np.zeros((h, w), np.uint8)
    maskimg[mask] =255
    
    ret, thresh = cv2.threshold(maskimg.astype('uint8'), 1, 255, 0)
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) > 0:
        # changed, len(contours) == many -> len(contours) == 1
        new_contours = []
        for cnt in contours:
            new_contours += [pt[0] for pt in cnt]
        new_contours = np.array(new_contours)
        hull = cv2.convexHull(new_contours)
        new_hull = []
        for h in hull:
            new_hull.append(h[0].tolist())
            
        return new_hull
    else:
        return None
    
def createPolygonData(label, points):
    labelme_polygon = {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
    return labelme_polygon

def main(search_path, save_path='./ContoursRes/'):
    if not os.path.isdir(search_path):
        print("Not Found...", search_path)
        return -1
    
    createFolder(save_path)
    filelists = search(search_path)
    
    for fullpath in filelists:
        imgname = fullpath.split('/')[-1]
        img = cv2.imread(fullpath)
        h, w, c = img.shape
        labelme_json = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": imgname,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w
        }

        labelme_polygon = {
            "label": "",
            "points": [],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }

        #FindColorList(img)

        peak_hull = createHullPoint(img, findcolor=[0, 0, 255])
        body_hull = createHullPoint(img, findcolor=[255, 0, 0])

        if peak_hull is not None:
            peak_labelme = createPolygonData("peak", peak_hull)
            labelme_json["shapes"].append(peak_labelme)
            
        if body_hull is not None:
            body_labelme = createPolygonData("body", body_hull)
            labelme_json["shapes"].append(body_labelme)

        with open(save_path + '/' + imgname.split('.')[0] + '.json', 'w') as f:
            json.dump(labelme_json, f, indent=2)


main("Sample/", save_path='./Sample/')