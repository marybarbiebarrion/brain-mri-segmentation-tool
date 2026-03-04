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

    with torch.no_grad():
        for slice_idx in range(image_volume.shape[2]):
            slice_2d = image_volume[:, :, slice_idx].astype(np.float32)
            if np.max(slice_2d) == 0: continue # Skip empty slices
                
            slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
            
            # Padding to 240x240 for ViT architectures
            pad_h = max(0, 240 - slice_norm.shape[0])
            pad_w = max(0, 240 - slice_norm.shape[1])
            if pad_h > 0 or pad_w > 0:
                 slice_norm = np.pad(slice_norm, ((0, pad_h), (0, pad_w)), mode='constant')
                 
            input_tensor = torch.tensor(slice_norm).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Monai models return a tuple/list in eval mode sometimes, handle cleanly
            seg_logits = model(input_tensor)
            if isinstance(seg_logits, (tuple, list)): seg_logits = seg_logits[0]
                
            seg_pred = torch.argmax(seg_logits, dim=1).cpu().squeeze().numpy()
            
            # Crop padding back to original image dimensions
            h, w = image_volume.shape[0], image_volume.shape[1]
            pred_volume[:, :, slice_idx] = seg_pred[:h, :w]
            
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
            selected_model = st.selectbox("1. Select Architecture", list(MODELS.keys()))
            st.markdown("<br>", unsafe_allow_html=True) 
            
            uploaded_mri = st.file_uploader("2. Upload MRI Scan (e.g., T1/T2)", type=["nii", "nii.gz"])
            uploaded_gt = st.file_uploader("3. Upload Ground Truth Mask (Optional)", type=["nii", "nii.gz"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Run Analysis", use_container_width=True):
                if uploaded_mri is not None:
                    with st.spinner(f"Processing 3D volume with {selected_model}..."):
                        st.session_state['mri_data'] = load_nifti_file(uploaded_mri)
                        st.session_state['pred_mask'] = run_model_inference_3d(st.session_state['mri_data'], selected_model)
                        if uploaded_gt is not None:
                            gt_data = load_nifti_file(uploaded_gt).astype(np.int64)
                            gt_data[gt_data == 4] = 3 # Map label 4 to 3
                            st.session_state['gt_data'] = gt_data
                        else:
                            st.session_state['gt_data'] = None
                    st.session_state['run'] = True
                else:
                    st.error("Please upload an MRI file first.")
                    st.session_state['run'] = False
                
        with col2:
            st.subheader("Interactive 3D Dashboard")
            if 'run' in st.session_state and st.session_state['run']:
                mri_vol, pred_vol, gt_vol = st.session_state['mri_data'], st.session_state['pred_mask'], st.session_state['gt_data']
                max_slice = mri_vol.shape[2] - 1
                slice_idx = st.slider("Navigate Brain Slices (Z-axis)", 0, max_slice, max_slice // 2)
                
                mri_slice, pred_slice = mri_vol[:, :, slice_idx], pred_vol[:, :, slice_idx]
                pred_slice_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
                
                num_plots = 3 if gt_vol is None else 4
                fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
                
                axes[0].imshow(mri_slice, cmap='gray')
                axes[0].set_title(f"Input MRI (Slice {slice_idx})", fontsize=14, pad=10); axes[0].axis('off')
                
                axes[1].imshow(mri_slice, cmap='gray')
                axes[1].imshow(pred_slice_masked, cmap=brats_cmap, vmin=0, vmax=3, alpha=0.6)
                axes[1].set_title("Predicted Overlay", fontsize=14, pad=10); axes[1].axis('off')
                
                if gt_vol is not None:
                    gt_slice_masked = np.ma.masked_where(gt_vol[:, :, slice_idx] == 0, gt_vol[:, :, slice_idx])
                    axes[2].imshow(mri_slice, cmap='gray')
                    axes[2].imshow(gt_slice_masked, cmap=brats_cmap, vmin=0, vmax=3, alpha=0.6)
                    axes[2].set_title("Ground Truth Overlay", fontsize=14, pad=10); axes[2].axis('off')
                    
                    axes[3].imshow(pred_slice_masked, cmap=brats_cmap, vmin=0, vmax=3, alpha=0.8)
                    axes[3].set_title("Prediction Isolated", fontsize=14, pad=10); axes[3].axis('off')
                else:
                    axes[2].imshow(pred_slice_masked, cmap=brats_cmap, vmin=0, vmax=3, alpha=0.8)
                    axes[2].set_title("Prediction Isolated", fontsize=14, pad=10); axes[2].axis('off')
                
                plt.tight_layout(); st.pyplot(fig)
                st.markdown("**Legend:** 🔴 NCR/NET &nbsp;&nbsp;|&nbsp;&nbsp; 🟢 Edema &nbsp;&nbsp;|&nbsp;&nbsp; 🔵 Enhancing Tumor")
            else:
                st.info("👈 Upload your T1/T2 scan, an optional segmentation mask, and click 'Run Analysis'.")

    elif page == "2. Model Information":
        st.title("Model Architectures & Metrics")
        st.table(MODEL_METRICS)

if __name__ == "__main__":
    main()