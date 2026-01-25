import os
import json
import numpy as np
import cv2
import random
from pycocotools import mask
from refer import REFER

w_new = h_new = 16
data_path = "./datasets/refer_seg"
dss = ["refcoco", "refcoco+", "refcocog", "refclef"]
Content = []
j = 0


QUESTION_TEMPLATES = [
    '"{q_placeholder}"\'s polygon',
    "What is the polygon of {q_placeholder} in the image?",
    "Output the polygon coordinates of {q_placeholder} in the image.",
    "Tell me the polygon coordinates of {q_placeholder} in the image.",
    "Give me the polygon coordinates of {q_placeholder} in the image.",
    "Help me locate the polygon of {q_placeholder}.",
    "Locate the polygon of {q_placeholder}.",
]

ANSWER_TEMPLATES = [
    "{a_placeholder}.",
    "{a_placeholder}.",
    "Sure, you can find it at {a_placeholder}.",
    "It is found at {a_placeholder}.",
    "It is at {a_placeholder}.",
]


def find_polygons(mask: np.ndarray, epsilon_ratio: float = 0.001) -> list[np.ndarray]:
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    for contour in contours:
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)  # 设置近似精度
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = approx.reshape(-1, 2)
        polygons.append(vertices)

    return polygons


for ds in dss:
    if ds == "refcocog":
        splitBy = "umd"
    else:
        splitBy = "unc"

    refer_api = REFER(data_path, ds, splitBy)

    ref_ids_train = refer_api.getRefIds(split="train")
    images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)  
    loaded_images = refer_api.loadImgs(image_ids=images_ids_train) 

    for _ in range(2):
        for num, image_info in zip(range(len(loaded_images)), loaded_images):
            refs = refer_api.imgToRefs[image_info["id"]]
            for ref in refs:
                if ds == "refclef":
                    img_path = os.path.join("./datasets", "refer_seg/images/saiapr_tc-12", image_info["file_name"])
                else:
                    img_path = os.path.join("./datasets", "refer_seg/images/coco_2014/train2014", image_info["file_name"])

                sentences = ref['sentences']
                ann = refer_api.refToAnn[ref['ref_id']]

                if type(ann["segmentation"][0]) == list:  # polygon
                    rle = mask.frPyObjects(
                        ann["segmentation"],
                        image_info["height"],
                        image_info["width"],
                    )
                else:
                    rle = ann["segmentation"]
                    for i in range(len(rle)):
                        if not isinstance(rle[i]["counts"], bytes):
                            rle[i]["counts"] = rle[i]["counts"].encode()
                m = mask.decode(rle)
                m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
                m[m > 1] = 1

                polygons = find_polygons(m)
                polygons = [
                    p / np.array([image_info["width"], image_info["height"]]) for p in polygons
                ]
                polygons = [np.round(p, 3) for p in polygons]
                polygons = [p.tolist() for p in polygons]

                for sentence in sentences:
                    item = {}
                    # conversations
                    conversation_list = []
                    question = random.choice(QUESTION_TEMPLATES).format(q_placeholder=sentence)

                    conversation_list.append({"from": "user", "value": f"<img>{img_path}</img>{question}"})

                    answer = random.choice(ANSWER_TEMPLATES).format(a_placeholder=polygons)

                    conversation_list.append({"from": "assistant", "value": answer})

                    item["conversations"] = conversation_list

                    Content.append(item)
                    j += 1
                    print(j)


with open("./train_data.json", "w") as f:
    json.dump(Content, f, indent=4)