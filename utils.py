from pathlib import Path
import cv2
from matplotlib import cm
import numpy as np
import pandas as pd
from GCVit import load_model
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image


model = load_model("gcvit_xxtiny", pretrained=True, num_classes=2)
model.load_state_dict(torch.load("gcvit-xxtiny-pneumonia-224.pth", weights_only=False))
model.cuda()
model.eval()


def load_gradcam_method():
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    def gradcam(image: torch.Tensor, img):
        # image: 1, 3, 224, 224
        sample_numpy = np.array(image[0][0].cpu())
        sample_rgb = np.repeat(sample_numpy[:, :, np.newaxis], 3, axis=2)
        grayscale_cam = cam(input_tensor=image)[0, :]  # 处理单张图像
        # 将热力图叠加到原始图像上
        img_np = np.array(sample_rgb)
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        return visualization

    return gradcam


cmap = cm.get_cmap("jet")
colors = cmap(np.arange(256))[:, :3]  # Get colormap


def load_gcvit_method():
    def global_attention(img_tensor: torch.Tensor, img):
        img_tensor.requires_grad = True
        feats = model.forward_features(img_tensor)  # Feature extraction
        feats.retain_grad()
        preds = model.forward_head(feats)  # Prediction head
        pred_index = preds.argmax(
            dim=-1
        ).item()  # Use highest probability class if no index provided
        class_channel = preds[
            :, pred_index
        ]  # Select the output corresponding to the chosen class
        model.zero_grad()  # Zero out previous gradients
        class_channel.backward(
            retain_graph=True
        )  # Backpropagate to get gradients w.r.t feats

        grads = feats.grad  # Get the gradient of the class score w.r.t feats
        pooled_grads = grads.mean(
            dim=(0, 2, 3)
        )  # Mean pooling across spatial dimensions

        heatmap = (feats[0] * pooled_grads[..., None, None]).sum(
            dim=0
        )  # Matrix multiplication for weighted sum of channels

        heatmap = torch.maximum(
            heatmap, torch.tensor(0)
        )  # Apply ReLU to retain positive values
        heatmap /= heatmap.max()  # Normalize to [0, 1]
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = np.uint8(255 * heatmap)  # Scale to [0, 255] for visualization

        heatmap_colored = colors[heatmap]
        heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize(
            (img_tensor.shape[2], img_tensor.shape[3])
        )  # Resize to original image size
        heatmap_array = np.array(heatmap_img)

        alpha = 0.3
        img_rgb = np.array(img)
        # img_rgb = np.stack([np.array(img)] * 3, axis=-1)  # Shape: (224, 224, 3)
        overlay = (img_rgb * (1 - alpha) + heatmap_array * alpha).astype(np.uint8)

        return overlay, preds.detach().cpu().numpy()

    return global_attention


class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(
        self,
        input,
        index=None,
        method="transformer_attribution",
        is_ablation=False,
        start_layer=0,
    ):
        output = self.model(input)
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(
            torch.tensor(one_hot_vector).to(input.device),
            method=method,
            is_ablation=is_ablation,
            start_layer=start_layer,
            **kwargs,
        )


attribution_generator = LRP(model)


def generate_visualization(original_image, class_index=None):
    original_image = original_image.squeeze(0)
    transformer_attribution = attribution_generator.generate_LRP(
        original_image.unsqueeze(0).cuda(),
        method="transformer_attribution",
        index=class_index,
    ).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution, scale_factor=16, mode="bilinear"
    )
    transformer_attribution = (
        transformer_attribution.reshape(224, 224).data.cpu().numpy()
    )
    transformer_attribution = (
        transformer_attribution - transformer_attribution.min()
    ) / (transformer_attribution.max() - transformer_attribution.min())

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (
        image_transformer_attribution - image_transformer_attribution.min()
    ) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = 255 - np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def vit_explainability(img_tensor: torch.Tensor, img):
    return generate_visualization(img_tensor, None)


def load_pneumonia_dataset():
    root_path = Path("dataset/PneumoniaMNIST/test/pneumoniamnist_224/")
    # grep all images
    images = list(root_path.glob("*.png"))
    # image file name: test21_1.png
    # label: 0 for normal, 1 for pneumonia
    # to dataframe
    image_names = [
        {
            "image_id": image.stem.split("_")[0].replace("test", ""),
            "label": "normal" if image.stem.split("_")[1] == "0" else "pneumonia",
        }
        for image in images
    ]
    df = pd.DataFrame(image_names)
    df = df.astype({"image_id": int})
    # order by image_id
    df = df.sort_values(by="image_id", ascending=True)
    df.reset_index(drop=True, inplace=True)
    return df
