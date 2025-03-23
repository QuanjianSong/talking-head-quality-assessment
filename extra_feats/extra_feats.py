from PIL import Image
from torchvision import transforms
import pyiqa
import torch
from pyiqa.archs.qalign_arch import QAlign
# from pyiqa.archs.topiq_arch import QAlign
from extra_feats.custom_pyiqa_component import create_metric
from extra_feats.custom_topiq_arch import CustomCFANet


def init_metric_model(metric_list, device):

    # Initialize the metric model dictionary
    metric_model_dict = {}

    for metric_name in metric_list:
        try:
            # Directly create the metric using the metric_name
            # metric_model_dict[metric_name] = pyiqa.create_metric(metric_name.lower(), device=device)
            metric_model_dict[metric_name] = create_metric(metric_name.lower(), device=device)
        except Exception as e:
            raise ValueError(f"Error initializing metric '{metric_name}': {e}")

    print("Metric models were all loaded!")

    return metric_model_dict


# class OurQAlign(Align):
#     def __init__(self, dtype='fp16') -> None:
#         super().__init__()


if __name__ in "__main__":
    # metric_list = ["qalign", "topiq_nr-face-custom"]
    metric_list = ["topiq_nr-face-custom"]
    image_path = "NTIRE25/extra_feats/test.jpg"
    device = "cuda"

    image = Image.open(image_path).convert("RGB")
    h, w = image.size
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Shape: [1, 3, H, W]
    image_tensor = image_tensor.repeat(8, 1, 1, 1)  # Shape: [8, 3, H, W]

    faceiqa_model = create_metric("topiq_nr-face-custom", device=device)

    feat = faceiqa_model(image_tensor)





