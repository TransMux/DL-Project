## Deep Learning Project

Global Attention + GradCam + Vit Explainability interpretions.

This study aims to enhance the accessibility of medical image analysis tools for patients by integrating explainable artificial intelligence models into medical imaging processing. We adapted the Global Context Vision Transformer (GC ViT) architecture to work with the MEDMNIST dataset, focusing specifically on the PneumoniaMNIST subset for pneumonia detection. We fine-tuned the model to better capture disease-specific features and implemented an interpretability framework that includes Grad-CAM, Global Attention visualization, and Transformer-specific explainability techniques. Experimental results show that GC ViT achieves strong performance in medical image classification tasks, even with limited data and computational resources. The explainability methods provided valuable insights into the model's decision-making process, enhancing transparency and trust in AI-assisted medical diagnostics.

## Install

To run this project, since the three papers span a wide range of years, you will need to use three separate environments to install the dependencies for each project. After training the models, convert them to the corresponding version formats and then load and run them.

## References

[1]. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2020). Grad-CAM: visual explanations from deep networks via gradient-based localization. International journal of computer vision, 128, 336-359.

[2]. Hatamizadeh, A., Yin, H., Heinrich, G., Kautz, J., & Molchanov, P. (2022). Global Context Vision Transformers.

[3]. Chefer, H., Gur, S., & Wolf, L. (2020). Transformer interpretability beyond attention visualization.