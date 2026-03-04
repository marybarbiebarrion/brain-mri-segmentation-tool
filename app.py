import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from matplotlib.colors import ListedColormap

# Import external architecture libraries for baselines
import segmentation_models_pytorch as smp
from monai.networks.nets import SwinUNETR, SegResNet

# ==============================================================================
# 1. CUSTOM ARCHITECTURE CLASSES (For MAE-B, DINO, CMAE, BYOL, MAE, UNET)
# ==============================================================================

class ViTEncoderForSegmentation(nn.Module):
    def __init__(self, in_channels=1, image_size=240, patch_size=16, emb_dim=192, num_layer=12, num_head=3):
        super().__init__()
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, emb_dim))
        self.patchify = nn.Conv2d(in_channels, emb_dim, patch_size, patch_size)
        self.transformer_blocks = nn.ModuleList([Block(emb_dim, num_head) for _ in range(num_layer)])
        self.init_weight()
    def init_weight(self): trunc_normal_(self.pos_embedding, std=.02)
    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> b (h w) c') + self.pos_embedding
        features = []
        x = patches
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            if i in [2, 5, 8, 11]: 
                features.append(rearrange(x, 'b (h w) c -> b c h w', h=img.shape[2]//self.patch_size))
        return features

class UNetDecoder(nn.Module):
    def __init__(self, emb_dim=192, num_classes=4):
        super().__init__()
        self.upconv1 = nn.ConvTranspose2d(emb_dim, 512, 2, 2)
        self.dec_block1 = self._make_block(emb_dim + 512, 512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec_block2 = self._make_block(emb_dim + 256, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec_block3 = self._make_block(emb_dim + 128, 128)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
            nn.Conv2d(128, 64, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(64), nn.ReLU(True), 
            nn.Conv2d(64, num_classes, 1)
        )
    def _make_block(self, i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True), 
            nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)
        )
    def forward(self, features):
        f1, f2, f3, f4 = features
        x = self.upconv1(f4); x = torch.cat([x, F.interpolate(f3, size=x.shape[2:])], 1); x = self.dec_block1(x)
        x = self.upconv2(x); x = torch.cat([x, F.interpolate(f2, size=x.shape[2:])], 1); x = self.dec_block2(x)
        x = self.upconv3(x); x = torch.cat([x, F.interpolate(f1, size=x.shape[2:])], 1); x = self.dec_block3(x)
        return self.final_up(x)

class MAE_Segmentation_Model(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x): 
        return self.decoder(self.encoder(x))


# ==============================================================================
# 2. STREAMLIT CONFIGURATION
# ==============================================================================

MODELS = {
    "MAE-B": "MAE-B.pt",
    "UNet++": "UNET++.pt",
    "Swin UNETR": "Swin-UNETR.pt",
    "SegResNet": "SegResNet.pt",
    "CMAE": "CMAE.pt",
    "DINO": "DINO.pt",
    "MAE": "MAE.pt",
    "Attention U-Net": "Attention_UNet.pt",
    "BYOL": "BYOL.pt"
}

MODEL_METRICS = {
    "SegResNet": {"DSC": "0.1533", "HD": "8.5154", "S_composite": "0.2600"},
    "Attention U-Net": {"DSC": "0.1601", "HD": "11.2444", "S_composite": "0.2437"},
    "UNet++": {"DSC": "0.1661", "HD": "12.4439", "S_composite": "0.2411"},
    "CMAE": {"DSC": "0.1321", "HD": "15.7839", "S_composite": "0.2012"},
    "Swin UNETR": {"DSC": "0.1582", "HD": "20.2751", "S_composite": "0.2005"},
    "BYOL": {"DSC": "0.1053", "HD": "13.7708", "S_composite": "0.1925"},
    "MAE-B": {"DSC": "0.1403", "HD": "18.9988", "S_composite": "0.1916"},
    "MAE": {"DSC": "0.1328", "HD": "18.3524", "S_composite": "0.1897"},
    "DINO": {"DSC": "0.1021", "HD": "18.4960", "S_composite": "0.1694"}
}

brats_cmap = ListedColormap(['none', 'red', 'limegreen', 'blue'])

@st.cache_data
def load_nifti_file(file_buffer):
    with open("temp.nii", "wb") as f:
        f.write(file_buffer.getbuffer())
    data = nib.load("temp.nii").get_fdata()
    return np.rot90(data, k=1, axes=(0, 1))

@st.cache_resource
def load_pytorch_model(model_name, device):
    """Dynamically builds the correct architecture based on the dropdown selection."""
    num_classes = 4 # 0: BG, 1: NCR, 2: Edema, 3: ET
    
    if model_name == "UNet++":
        model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=num_classes)
        
    elif model_name == "Swin UNETR":
        model = SwinUNETR(spatial_dims=2, in_channels=1, out_channels=num_classes, feature_size=24, use_checkpoint=True)
        
    elif model_name == "SegResNet":
        model = SegResNet(spatial_dims=2, in_channels=1, out_channels=num_classes, init_filters=32, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], dropout_prob=0.2)
        
    else:
        # Default thesis architecture for MAE-B, MAE, DINO, CMAE, BYOL, Attention U-Net
        seg_encoder = ViTEncoderForSegmentation(image_size=240, patch_size=16, emb_dim=192, num_layer=12, num_head=3)
        seg_decoder = UNetDecoder(emb_dim=192, num_classes=num_classes)
        model = MAE_Segmentation_Model(seg_encoder, seg_decoder)

    model = model.to(device)
    weight_file = MODELS.get(model_name)
    
    try:
        model.load_state_dict(torch.load(weight_file, map_location=device))
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Could not load weights for {model_name}. Please ensure {weight_file} is in the same directory.")
        return None

def run_model_inference_3d(image_volume, model_name):
    """Runs the selected model across the Z-axis of the MRI volume."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_pytorch_model(model_name, device)
    pred_volume = np.zeros_like(image_volume, dtype=np.uint8)
    
    if model is None:
        return pred_volume

    # Define models that strictly require 256x256 inputs
    requires_padding = model_name in ["UNet++", "Swin UNETR", "SegResNet"]

    with torch.no_grad():
        for slice_idx in range(image_volume.shape[2]):
            slice_2d = image_volume[:, :, slice_idx].astype(np.float32)
            if np.max(slice_2d) == 0: continue # Skip empty background slices
                
            # Normalize the slice
            slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
            original_h, original_w = slice_norm.shape[0], slice_norm.shape[1]
            
            # --- CONDITIONAL PADDING FOR BASELINE MODELS ---
            if requires_padding:
                # Pad from 240x240 to 256x256 (adding 8 pixels on all sides)
                pad_h = max(0, 256 - original_h)
                pad_w = max(0, 256 - original_w)
                # Pad equally on both sides: (left, right, top, bottom)
                pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)
                pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
                
                slice_norm = np.pad(slice_norm, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
            else:
                # Standard padding to 240x240 for custom ViT models if needed
                pad_h = max(0, 240 - original_h)
                pad_w = max(0, 240 - original_w)
                if pad_h > 0 or pad_w > 0:
                     slice_norm = np.pad(slice_norm, ((0, pad_h), (0, pad_w)), mode='constant')
                 
            input_tensor = torch.tensor(slice_norm).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Inference
            seg_logits = model(input_tensor)
            if isinstance(seg_logits, (tuple, list)): seg_logits = seg_logits[0]
                
            # Get the predicted class (0 to 3)
            seg_pred = torch.argmax(seg_logits, dim=1).cpu().squeeze().numpy()
            
            # --- CROPPING BACK TO ORIGINAL SIZE ---
            if requires_padding:
                # Remove the 8-pixel border we added
                seg_pred = seg_pred[pad_top : pad_top + original_h, pad_left : pad_left + original_w]
            else:
                seg_pred = seg_pred[:original_h, :original_w]
                
            pred_volume[:, :, slice_idx] = seg_pred
            
    return pred_volume

def main():
    st.set_page_config(page_title="Brain MRI Segmenter", layout="wide", initial_sidebar_state="expanded")
    st.sidebar.title("Navigation Menu")
    page = st.sidebar.radio("Go to:", ["1. Workspace", "2. Model Information"])
    
    if page == "1. Workspace":
        st.title("Brain MRI Segmentation Workspace")
        st.markdown("Upload a 3D NIfTI scan to visualize tumor regions using deep learning architectures.")
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2.5])
        with col1:
            st.subheader("Controls")
            
            # Feature: Multi-model selection (Max 3)
            compare_mode = st.toggle("Enable Model Comparison")
            
            if compare_mode:
                selected_models = st.multiselect(
                    "Select up to 3 Models to Compare", 
                    options=list(MODELS.keys()), 
                    max_selections=3
                )
            else:
                # Standard single model mode
                selected_model = st.selectbox("Select Architecture", list(MODELS.keys()))
                selected_models = [selected_model]
            
            st.markdown("<br>", unsafe_allow_html=True) 
            
            uploaded_mri = st.file_uploader("Upload MRI Scan (e.g., T1/T2)", type=["nii", "nii.gz"])
            uploaded_gt = st.file_uploader("Upload Ground Truth Mask (Optional)", type=["nii", "nii.gz"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Run Analysis", use_container_width=True):
                if uploaded_mri is not None and len(selected_models) > 0:
                    with st.spinner(f"Processing 3D volume with {len(selected_models)} model(s)..."):
                        # Load MRI once
                        st.session_state['mri_data'] = load_nifti_file(uploaded_mri)
                        
                        # Dictionary to store multiple predictions
                        st.session_state['all_preds'] = {}
                        for model_name in selected_models:
                            st.session_state['all_preds'][model_name] = run_model_inference_3d(
                                st.session_state['mri_data'], model_name
                            )
                        
                        if uploaded_gt is not None:
                            gt_data = load_nifti_file(uploaded_gt).astype(np.int64)
                            gt_data[gt_data == 4] = 3 
                            st.session_state['gt_data'] = gt_data
                        else:
                            st.session_state['gt_data'] = None
                            
                    st.session_state['run'] = True
                    st.session_state['active_models'] = selected_models
                elif uploaded_mri is None:
                    st.error("Please upload an MRI file first.")
                else:
                    st.error("Please select at least one model.")
                
        with col2:
            st.subheader("Interactive Comparison Dashboard")
            if 'run' in st.session_state and st.session_state['run']:
                mri_vol = st.session_state['mri_data']
                gt_vol = st.session_state['gt_data']
                active_models = st.session_state['active_models']
                
                max_slice = mri_vol.shape[2] - 1
                slice_idx = st.slider("Navigate Brain Slices (Z-axis)", 0, max_slice, max_slice // 2)
                
                brats_cmap = ListedColormap(['none', 'red', 'limegreen', 'blue'])
                mri_slice = mri_vol[:, :, slice_idx]
                
                # Dynamic column logic
                # Order: [Original] + [GT (if exists)] + [Model 1] + [Model 2 (optional)] + [Model 3 (optional)]
                plot_titles = ["Original Image"]
                if gt_vol is not None:
                    plot_titles.append("Ground Truth")
                for m_name in active_models:
                    plot_titles.append(f"Pred: {m_name}")
                
                num_plots = len(plot_titles)
                fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
                
                # Handle single plot case (axes won't be an array)
                if num_plots == 1:
                    axes = [axes]
                
                # Plot 1: Original Image
                axes[0].imshow(mri_slice, cmap='gray')
                axes[0].set_title(f"{plot_titles[0]} ({slice_idx})", fontsize=12, pad=10)
                axes[0].axis('off')
                
                current_col = 1
                
                # Plot 2: Ground Truth (Optional)
                if gt_vol is not None:
                    gt_slice_masked = np.ma.masked_where(gt_vol[:, :, slice_idx] == 0, gt_vol[:, :, slice_idx])
                    axes[current_col].imshow(mri_slice, cmap='gray')
                    axes[current_col].imshow(gt_slice_masked, cmap=brats_cmap, vmin=0, vmax=3, alpha=0.6)
                    axes[current_col].set_title(plot_titles[current_col], fontsize=12, pad=10)
                    axes[current_col].axis('off')
                    current_col += 1
                
                # Remaining Plots: Selected Model Predictions
                for model_name in active_models:
                    pred_vol = st.session_state['all_preds'][model_name]
                    pred_slice = pred_vol[:, :, slice_idx]
                    pred_slice_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
                    
                    axes[current_col].imshow(mri_slice, cmap='gray')
                    axes[current_col].imshow(pred_slice_masked, cmap=brats_cmap, vmin=0, vmax=3, alpha=0.6)
                    axes[current_col].set_title(plot_titles[current_col], fontsize=12, pad=10)
                    axes[current_col].axis('off')
                    current_col += 1
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("Legend: Red: NCR/NET | Green: Edema | Blue: Enhancing Tumor")
                
                # Display Metrics Scoreboard for active models
                st.markdown("### Model Comparison Scoreboard")
                m_cols = st.columns(len(active_models))
                for idx, m_name in enumerate(active_models):
                    with m_cols[idx]:
                        st.markdown(f"**{m_name}**")
                        metrics = MODEL_METRICS.get(m_name, {})
                        st.write(f"DSC: {metrics.get('DSC', 'N/A')}")
                        st.write(f"HD: {metrics.get('HD', 'N/A')} mm")
            else:
                st.info("Upload your scan and select up to 3 models to begin comparison.")

    elif page == "2. Model Information":
        st.title("Model Selection Scoreboard")
        st.markdown("Use this guide to determine which architecture is best suited for your specific clinical scenario.")
        
        # [Keep existing Scoreboard columns and sorted table here]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("Best Volumetric Overlap\n\n### UNet++\n**DSC: 0.1661**\n\nScenario: When capturing the bulk mass and overall volume of the tumor is the highest priority.")
        with col2:
            st.success("Best Boundary Precision\n\n### SegResNet\n**HD: 8.51 mm**\n\nScenario: For surgical planning where detecting exact tumor edges is critical.")
        with col3:
            st.warning("Best Overall Balance\n\n### SegResNet\n**S_comp: 0.2600**\n\nScenario: Reliable all-rounder balancing volume detection and boundary precision.")

        st.markdown("---")
        sorted_metrics = dict(sorted(MODEL_METRICS.items(), key=lambda item: float(item[1]['DSC']), reverse=True))
        st.table(sorted_metrics)

if __name__ == "__main__":
    main()