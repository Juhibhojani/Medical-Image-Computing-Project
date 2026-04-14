"""Implements feature visualization tracking while student learns via feature distillation"""

import torch
import os
from typing import List
import matplotlib.pyplot as plt
import torch.nn as nn


class FeatureTracker:
    def __init__(self, teacher_model, student_model, device,teacher_layers: List[int] = [1, 2, 3], student_layers: List[int] =[1, 2, 3], save_dir="feature_visualization_logs"):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_layers = teacher_layers
        self.student_layers = student_layers
        self.device = device

        self.teacher_features = []
        self.student_features = []
        self.inputs = None

        # making directory to save results
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def capture(self, inputs):
        """Capture features for ONE fixed batch"""
        self.teacher_features = []
        self.student_features = []
        self.inputs = inputs

        with torch.no_grad():
            # logic is same as feature distillation
            for t_layer, s_layer in zip(self.teacher_layers, self.student_layers):
                teacher_layers=nn.Sequential(*list(self.teacher_model.children())[:t_layer])
                student_layers=nn.Sequential(*list(self.student_model.children())[:s_layer])
                
                teacher_feature=teacher_layers(inputs)
                student_feature=student_layers(inputs)
                
                self.teacher_features.append(teacher_feature.detach().cpu())
                self.student_features.append(student_feature.detach().cpu())

    def save_visualization(self, epoch):
        """Save feature maps + difference maps"""
        # iterating for all 3 layers
        for idx, (t_feat, s_feat) in enumerate(zip(self.teacher_features, self.student_features)):

            # using only feature map for first sample
            t_feat = t_feat[0] 
            s_feat = s_feat[0]
            input_img = self.inputs[0].detach().cpu()
            
            mse_value = torch.nn.functional.mse_loss(t_feat, s_feat).item()

            # convert CHW → HWC
            input_img = input_img.permute(1, 2, 0)

            # normalize input for display
            input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-5)

            num_channels = min(5, t_feat.shape[0])

            fig, axes = plt.subplots(4, num_channels, figsize=(15, 8))

            for i in range(num_channels):
                axes[0, i].imshow(input_img)
                axes[0, i].set_title("Input")
                axes[0, i].axis('off')

                axes[1, i].imshow(t_feat[i], cmap='gray')
                axes[1, i].set_title(f"T-{i}")
                axes[1, i].axis('off')

                axes[2, i].imshow(s_feat[i], cmap='gray')
                axes[2, i].set_title(f"S-{i}")
                axes[2, i].axis('off')
                
                diff = torch.abs(t_feat[i] - s_feat[i])
                axes[3, i].imshow(diff, cmap='jet')
                axes[3, i].set_title("Diff")
                axes[3, i].axis('off')

            fig.suptitle(f"Epoch {epoch} | Layer {idx} | MSE: {mse_value:.6f}", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/epoch_{epoch}_layer_{idx}.png")
            plt.close()
                
                