import gradio as gr
import numpy as np
from torchvision import transforms
from utils import (
    load_gcvit_method,
    load_pneumonia_dataset,
    load_gradcam_method,
    vit_explainability,
)
from PIL import Image

# load model
gradcam = load_gradcam_method()

global_attention = load_gcvit_method()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ]
)

data = load_pneumonia_dataset()


def handle_table_click(selected_data: gr.SelectData):
    index = selected_data.index[0]  # 从 SelectData 中提取选定的行索引
    row = data.iloc[index]
    label_int = 0 if row["label"] == "normal" else 1
    return "./dataset/PneumoniaMNIST/test/pneumoniamnist_224/test{}_{}.png".format(
        row["image_id"], label_int
    )


def predict_fn(sample):
    if not sample:
        return None, None, None, None

    sample = Image.open(sample)
    sample_tensor = transform(sample).cuda()
    sample_tensor = sample_tensor.unsqueeze(0)

    global_attention_output, preds = global_attention(sample_tensor, sample)

    print(preds)

    sample_tensor.requires_grad = False

    probabilities = 1 / (1 + np.exp(-preds))

    return (
        global_attention_output,
        gradcam(sample_tensor, sample),
        vit_explainability(sample_tensor, sample),
        {"Pneumonia": probabilities[0][1], "Normal": probabilities[0][0]},
    )


with gr.Blocks(
    theme=gr.themes.Soft(), title="MSAI - Deep Learning Group Project"
) as demo:
    gr.Markdown(
        value="""
# MSAI - Deep Learning Group Project

基于 MedMNIST 数据集的可解释性分析

- 使用最新的 GCVit 模型

- 针对同一个模型的预测结果，同时使用 Global Attention<sup>[1]</sup> 方法，GradCam<sup>[2]</sup> 方法，以及 Vit Explainability<sup>[3]</sup> 方法进行可解释性分析，以探求模型做出预测的原因，帮助改进模型效果
                """
    )

    with gr.Row(equal_height=True):
        input_image = gr.Image(label="Input Image", sources=["upload"], type="filepath")
        preds_label = gr.Label(value=None)

    with gr.Row():
        global_attention_output = gr.Image(label="Global Attention", interactive=False)
        gradcam_output = gr.Image(label="GradCAM", interactive=False)
        vit_explainability_output = gr.Image(
            label="Vit Explainability", interactive=False
        )

    input_image.change(
        predict_fn,
        inputs=input_image,
        outputs=[
            global_attention_output,
            gradcam_output,
            vit_explainability_output,
            preds_label,
        ],
    )

    gr.Markdown("## Data Input")

    table = gr.DataFrame(data, label="Test Dataset(624 images)", interactive=True)

    table.select(handle_table_click, outputs=input_image)

    gr.Markdown(
        """## References
                
[1]. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2020). Grad-CAM: visual explanations from deep networks via gradient-based localization. International journal of computer vision, 128, 336-359.

[2]. Hatamizadeh, A., Yin, H., Heinrich, G., Kautz, J., & Molchanov, P. (2022). Global Context Vision Transformers.

[3]. Chefer, H., Gur, S., & Wolf, L. (2020). Transformer interpretability beyond attention visualization.
                """
    )

demo.launch(
    allowed_paths=[
        "/root/projects/MSAI-DL-Project/dataset/PneumoniaMNIST/test/pneumoniamnist_224/",
        "./dataset/",
        "dataset",
    ]
)
