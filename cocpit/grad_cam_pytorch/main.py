import torch
import cocpit
import os
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt

from cocpit.grad_cam_pytorch.grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

from cocpit.grad_cam_pytorch.save_image import (
    save_gradient,
    save_gradcam,
    save_sensitivity,
)


class Process:
    def __init__(self, image_path):
        self.image_path = image_path
        self.raw_image = None

    def show_raw_image(self):
        self.raw_image = cv2.imread(self.image_path)
        self.raw_image = cv2.resize(self.raw_image, (224,) * 2)
        fig, (ax) = plt.subplots(
            constrained_layout=True, figsize=(5, 5), ncols=1, nrows=1
        )
        ax.imshow(self.raw_image)

    def rgb(self):
        self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)

    def transform(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transform(self.raw_image)


class Runner:
    def __init__(
        self, model, raw_image, image, topk, ids, probs, output_dir, target_layer
    ):
        self.model = model
        self.raw_image = raw_image
        self.image = image
        self.topk = topk
        self.ids = ids
        self.classes = cocpit.config.CLASS_NAMES
        self.output_dir = output_dir
        self.probs = probs
        self.output_dir = output_dir
        self.target_layer = target_layer

    def deconv(self):
        print("Deconvolution:")

        deconv = Deconvnet(model=self.model)
        _ = deconv.forward(self.image)

        for i in range(self.topk):
            deconv.backward(ids=self.ids[:, [i]])
            gradients = deconv.generate()

            print("\t{} ({:.5f})".format(self.classes[self.ids[i]], self.probs[i]))
            save_gradient(
                filename=os.path.join(
                    self.output_dir, f"deconvnet-{self.classes[self.ids[i]]}.png"
                ),
                gradient=gradients,
            )
        deconv.remove_hook()

    def gradcam(self):
        print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

        gcam = GradCAM(model=self.model)
        _ = gcam.forward(self.image)

        gbp = GuidedBackPropagation(model=self.model)
        _ = gbp.forward(self.image)

        for i in range(self.topk):
            # Guided Backpropagation
            gbp.backward(ids=self.ids[:, [i]])
            gradients = gbp.generate()

            # Grad-CAM
            gcam.backward(ids=self.ids[:, [i]])
            regions = gcam.generate(target_layer=self.target_layer)

            print("\t{} ({:.5f})".format(self.classes[self.ids[i]], self.probs[i]))

            # Guided Backpropagation
            save_gradient(
                filename=os.path.join(
                    self.output_dir, f"-guided-{self.classes[self.ids[i]]}.png"
                ),
                gradient=gradients[j],
            )

            # Grad-CAM
            save_gradcam(
                filename=os.path.join(
                    self.output_dir,
                    f"-gradcam-{self.target_layer}-{self.classes[self.ids[i]]}.png",
                ),
                gcam=regions[0],
                raw_image=self.raw_image,
            )

            # Guided Grad-CAM
            save_gradient(
                filename=os.path.join(
                    self.output_dir,
                    f"-guided_gradcam-{self.target_layer}-{self.classes[self.ids[i]]}.png",
                ),
                gradient=torch.mul(regions, gradients),
            )

    def vanilla_bp(self, bp):
        print("Vanilla Backpropagation:")

        for i in range(self.topk):
            bp.backward(ids=self.ids[:, [i]])
            gradients = bp.generate()

            # Save results as image files
            print("\t{} ({:.5f})".format(self.classes[self.ids[i]], self.probs[i]))
            save_gradient(
                filename=os.path.join(
                    self.output_dir, f"vanilla-{self.classes[self.ids[i]]}.png"
                ),
                gradient=gradients,
            )

        # Remove all the hook function in the "model"
        bp.remove_hook()


def main(image_path, model, target_layer, topk=3, output_dir="./results"):
    """
    Visualize model responses given multiple image
    """

    p = Process(image_path)
    p.show_raw_image()
    p.rgb()
    image = p.transform()

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(image)  # sorted
    print(ids)

    r = Runner(model, p.raw_image, image, topk, ids, probs, output_dir, target_layer)
    r.deconv()
    r.gradcam()
    r.vanilla_bp(bp)


if __name__ == "__main__":
    main()
