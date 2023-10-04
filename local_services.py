import copy
import os

import numpy as np
import tiktoken
import torch
from PIL import Image
from flask import Flask, request, abort
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from waitress import serve

from utils import plot_boxes_to_image

app = Flask(__name__)


@app.post("/prompt_len")
def prompt_len():
    if request.is_json:
        content = request.get_json()
        count = get_prompt_len(content["prompt"])
        res = {
            "token_count": count
        }
        return res
    return {"error": "Request must be JSON"}


@app.post("/owl_vit")
def owl_vit():
    # print(request.json)
    if not request.json or 'image' not in request.json:
        abort(400)
    im_array_from_request = np.array(request.json['image'], dtype=np.uint8)
    image = Image.fromarray(im_array_from_request, "RGB")
    image.save("outputs/from_request.jpeg")
    texts = [request.json['texts']]

    with torch.no_grad():
        inputs = owl_vit_processor(text=texts, images=image, return_tensors="pt").to(device)
        outputs = owl_vit_model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = owl_vit_processor.post_process_object_detection(outputs=outputs, threshold=0.0,
                                                              target_sizes=target_sizes.to(device))
    scores = torch.sigmoid(outputs.logits)
    topk_scores, topk_idxs = torch.topk(scores, k=1, dim=1)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    print("get_topk is true")
    topk_idxs = topk_idxs.squeeze(1).tolist()
    topk_boxes = results[i]['boxes'][topk_idxs]
    topk_scores = topk_scores.view(len(text), -1)
    topk_labels = results[i]["labels"][topk_idxs]
    boxes, scores, labels = topk_boxes, topk_scores, topk_labels

    # Print detected objects and rescaled box coordinates
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

    boxes = boxes.cpu().detach().numpy()
    normalized_boxes = copy.deepcopy(boxes)

    # # visualize pred
    size = image.size
    pred_dict = {
        "boxes": normalized_boxes,
        "size": [size[1], size[0]],  # H, W
        "labels": [text[idx] for idx in labels]
    }

    image_with_box = plot_boxes_to_image(image, pred_dict)[0]
    image_with_box.save(os.path.join("outputs/owlvit_box.jpg"))

    result_dict = {'output': 'ok'}
    return result_dict


def get_prompt_len(prompt):
    enc = tiktoken.get_encoding("cl100k_base")
    res = len(enc.encode(prompt))
    return res


if __name__ == "__main__":
    owlvit_model = "owlvit-base-patch32"
    device = "cuda:1"
    owl_vit_processor = OwlViTProcessor.from_pretrained(f"google/{owlvit_model}")
    owl_vit_model = OwlViTForObjectDetection.from_pretrained(f"google/{owlvit_model}")
    owl_vit_model.to(device)
    owl_vit_model.eval()

    serve(app, host="localhost", port=8081)
