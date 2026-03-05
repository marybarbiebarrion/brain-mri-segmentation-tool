import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from matplotlib.colors import ListedColormap

# Import external architecture libraries for baselines
import segmentation_models_pytorch as smp
from monai.networks.nets import SwinUNETR, SegResNet

# ==============================================================================
# 1. CUSTOM ARCHITECTURE CLASSES (100% SEPARATED FOR FAIRNESS)
# ==============================================================================

# --- ENCODERS ---
class MAE_B_Encoder(nn.Module):
    def __init__(self, in_channels=1, image_size=240, patch_size=16, emb_dim=192, num_layer=12, num_head=3):
        super().__init__()
        self.patchify = nn.Conv2d(in_channels, emb_dim, patch_size, patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, emb_dim))
        self.transformer_blocks = nn.ModuleList([Block(emb_dim, num_head) for _ in range(num_layer)])
        self.init_weight()
    def init_weight(self): trunc_normal_(self.pos_embedding, std=.02)
    def forward(self, img):
        x = self.patchify(img); x = rearrange(x, 'b c h w -> b (h w) c') + self.pos_embedding
        features = []
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            if i in [2, 5, 8, 11]: features.append(rearrange(x, 'b (h w) c -> b c h w', h=img.shape[2]//16))
        return features

class Standard_MAE_Encoder(nn.Module):
    def __init__(self, in_channels=1, image_size=240, patch_size=16, emb_dim=192, num_layer=12, num_head=3):
        super().__init__()
        self.patchify = nn.Conv2d(in_channels, emb_dim, patch_size, patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, emb_dim))
        self.transformer_blocks = nn.ModuleList([Block(emb_dim, num_head) for _ in range(num_layer)])
        self.init_weight()
    def init_weight(self): trunc_normal_(self.pos_embedding, std=.02)
    def forward(self, img):
        x = self.patchify(img); x = rearrange(x, 'b c h w -> b (h w) c') + self.pos_embedding
        features = []
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            if i in [2, 5, 8, 11]: features.append(rearrange(x, 'b (h w) c -> b c h w', h=img.shape[2]//16))
        return features

class DINO_CMAE_Encoder(nn.Module):
    def __init__(self, in_channels=1, image_size=240, patch_size=16, emb_dim=192, num_layer=12):
        super().__init__()
        self.patchify = nn.Conv2d(in_channels, emb_dim, patch_size, patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, emb_dim))
        self.blocks = nn.ModuleList([Block(emb_dim, 3) for _ in range(num_layer)])
    def forward(self, img):
        x = self.patchify(img); x = rearrange(x, 'b c h w -> b (h w) c') + self.pos_embedding
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [5, 7, 9, 11]: features.append(rearrange(x, 'b (h w) c -> b c h w', h=img.shape[2]//16))
        return features

# --- DECODER ---
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
        return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True), nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
    def forward(self, features):
        f1, f2, f3, f4 = features
        x = self.upconv1(f4); x = torch.cat([x, F.interpolate(f3, size=x.shape[2:])], 1); x = self.dec_block1(x)
        x = self.upconv2(x); x = torch.cat([x, F.interpolate(f2, size=x.shape[2:])], 1); x = self.dec_block2(x)
        x = self.upconv3(x); x = torch.cat([x, F.interpolate(f1, size=x.shape[2:])], 1); x = self.dec_block3(x)
        return self.final_up(x)

# --- INDIVIDUAL WRAPPERS ---
class MAEB_Wrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__(); self.encoder = encoder; self.decoder = decoder
    def forward(self, x): return self.decoder(self.encoder(x))

class MAE_Wrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__(); self.encoder = encoder; self.decoder = decoder
    def forward(self, x): return self.decoder(self.encoder(x))

class DINO_Wrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__(); self.encoder = encoder; self.decoder = decoder
    def forward(self, x): return self.decoder(self.encoder(x))

class CMAE_Wrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__(); self.encoder = encoder; self.decoder = decoder
    def forward(self, x): return self.decoder(self.encoder(x))

class BYOL_Wrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__(); self.encoder = encoder; self.decoder = decoder
    def forward(self, x): return self.decoder(self.encoder(x))

class AttnUNet_Wrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__(); self.encoder = encoder; self.decoder = decoder
    def forward(self, x): return self.decoder(self.encoder(x))

# ==============================================================================
# 2. STREAMLIT CONFIGURATION & METRICS
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
    """Dynamically builds the correct architecture with strict model separation."""
    num_classes = 4 
    
    # 1. BASELINE MODELS
    if model_name == "UNet++":
        model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=num_classes)
    elif model_name == "Swin UNETR":
        model = SwinUNETR(spatial_dims=2, in_channels=1, out_channels=num_classes, feature_size=24, use_checkpoint=True)
    elif model_name == "SegResNet":
        model = SegResNet(spatial_dims=2, in_channels=1, out_channels=num_classes, init_filters=32, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], dropout_prob=0.2)
        
    # 2. THESIS MODELS
    elif model_name == "MAE-B":
        model = MAEB_Wrapper(MAE_B_Encoder(), UNetDecoder(num_classes=num_classes))
    elif model_name == "MAE":
        model = MAE_Wrapper(Standard_MAE_Encoder(), UNetDecoder(num_classes=num_classes))
    elif model_name == "DINO":
        model = DINO_Wrapper(DINO_CMAE_Encoder(), UNetDecoder(num_classes=num_classes))
    elif model_name == "CMAE":
        model = CMAE_Wrapper(DINO_CMAE_Encoder(), UNetDecoder(num_classes=num_classes))
    elif model_name == "BYOL":
        model = BYOL_Wrapper(MAE_B_Encoder(), UNetDecoder(num_classes=num_classes))
    elif model_name == "Attention U-Net":
        model = AttnUNet_Wrapper(MAE_B_Encoder(), UNetDecoder(num_classes=num_classes))
    else:
        st.error("Model architecture not found.")
        return None

    model = model.to(device)
    weight_file = MODELS.get(model_name)
    
    try:
        model.load_state_dict(torch.load(weight_file, map_location=device), strict=True)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Could not load weights for {model_name}. Please ensure {weight_file} matches the exact architecture. Error: {e}")
        return None

def run_model_inference_3d(image_volume, model_name):
    """Runs high-fidelity inference mapping to spatial capabilities."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_pytorch_model(model_name, device)
    pred_volume = np.zeros_like(image_volume, dtype=np.uint8)
    
    if model is None: return pred_volume

    target_size = 256 if model_name in ["UNet++", "Swin UNETR", "SegResNet"] else 240

    with torch.no_grad():
        for slice_idx in range(image_volume.shape[2]):
            slice_2d = image_volume[:, :, slice_idx].astype(np.float32)
            if np.max(slice_2d) == 0: continue 
                
            s_min, s_max = slice_2d.min(), slice_2d.max()
            slice_norm = (slice_2d - s_min) / (s_max - s_min + 1e-8)
            
            orig_h, orig_w = slice_norm.shape
            pad_h, pad_w = max(0, target_size - orig_h), max(0, target_size - orig_w)
            p_top, p_left = pad_h // 2, pad_w // 2
            slice_norm = np.pad(slice_norm, ((p_top, pad_h-p_top), (p_left, pad_w-p_left)), mode='constant')
                 
            input_t = torch.tensor(slice_norm).float().unsqueeze(0).unsqueeze(0).to(device)
            
            output = model(input_t)
            if isinstance(output, (tuple, list)): output = output[0]
            seg_pred = torch.argmax(output, dim=1).cpu().squeeze().numpy()
            
            pred_volume[:, :, slice_idx] = seg_pred[p_top : p_top + orig_h, p_left : p_left + orig_w]
            
    return pred_volume

# ==============================================================================
# 3. MAIN APP EXECUTION
# ==============================================================================

def main():
    # Prevent KeyErrors by ensuring session state variables exist
    if 'run' not in st.session_state: st.session_state['run'] = False
    if 'mri_data' not in st.session_state: st.session_state['mri_data'] = None
    if 'gt_data' not in st.session_state: st.session_state['gt_data'] = None
    if 'all_preds' not in st.session_state: st.session_state['all_preds'] = {}
    if 'active_models' not in st.session_state: st.session_state['active_models'] = []
    if 'execution_time' not in st.session_state: st.session_state['execution_time'] = 0.0

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
            compare_mode = st.toggle("Enable Model Comparison")
            
            if compare_mode:
                selected_models = st.multiselect("Select up to 3 Models to Compare", options=list(MODELS.keys()), max_selections=3)
            else:
                selected_models = [st.selectbox("Select Architecture", list(MODELS.keys()))]
            
            st.markdown("<br>", unsafe_allow_html=True) 
            uploaded_mri = st.file_uploader("Upload MRI Scan (e.g., T1/T2)", type=["nii", "nii.gz"])
            uploaded_gt = st.file_uploader("Upload Ground Truth Mask (Optional)", type=["nii", "nii.gz"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Run Analysis", use_container_width=True):
                if uploaded_mri is not None and len(selected_models) > 0:
                    start_time = time.time()
                    with st.spinner(f"Processing 3D volume with {len(selected_models)} model(s)..."):
                        st.session_state['mri_data'] = load_nifti_file(uploaded_mri)
                        st.session_state['all_preds'] = {}
                        for model_name in selected_models:
                            st.session_state['all_preds'][model_name] = run_model_inference_3d(st.session_state['mri_data'], model_name)
                        
                        if uploaded_gt is not None:
                            gt_data = load_nifti_file(uploaded_gt).astype(np.int64)
                            gt_data[gt_data == 4] = 3 
                            st.session_state['gt_data'] = gt_data
                        else:
                            st.session_state['gt_data'] = None
                            
                    st.session_state['execution_time'] = time.time() - start_time
                    st.session_state['run'] = True
                    st.session_state['active_models'] = selected_models
                elif uploaded_mri is None:
                    st.error("Please upload an MRI file first.")
                else:
                    st.error("Please select at least one model.")
                
        with col2:
            st.subheader("Interactive Comparison Dashboard")
            if st.session_state['run']:
                total_time = st.session_state['execution_time']
                num_models = len(st.session_state['active_models'])
                
                t_col1, t_col2 = st.columns(2)
                t_col1.metric("Total Processing Time", f"{total_time:.2f} s")
                t_col2.metric("Average Time per Model", f"{total_time / num_models if num_models > 0 else 0:.2f} s")

                mri_vol = st.session_state['mri_data']
                gt_vol = st.session_state['gt_data']
                active_models = st.session_state['active_models']
                
                max_slice = mri_vol.shape[2] - 1
                slice_idx = st.slider("Navigate Brain Slices (Z-axis)", 0, max_slice, max_slice // 2)
                
                mri_slice = mri_vol[:, :, slice_idx]
                
                plot_titles = ["Original Image"]
                if gt_vol is not None: plot_titles.append("Ground Truth")
                for m_name in active_models: plot_titles.append(f"Pred: {m_name}")
                
                num_plots = len(plot_titles)
                fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
                if num_plots == 1: axes = [axes]
                
                axes[0].imshow(mri_slice, cmap='gray')
                axes[0].set_title(f"Original Image ({slice_idx})", fontsize=12, pad=10)
                axes[0].axis('off')
                
                current_col = 1
                if gt_vol is not None:
                    gt_mask = np.ma.masked_where(gt_vol[:, :, slice_idx] == 0, gt_vol[:, :, slice_idx])
                    axes[current_col].imshow(mri_slice, cmap='gray')
                    axes[current_col].imshow(gt_mask, cmap=brats_cmap, vmin=0, vmax=3, alpha=0.6)
                    axes[current_col].set_title("Ground Truth", fontsize=12, pad=10)
                    axes[current_col].axis('off')
                    current_col += 1
                
                for model_name in active_models:
                    pred_slice = st.session_state['all_preds'][model_name][:, :, slice_idx]
                    pred_mask = np.ma.masked_where(pred_slice == 0, pred_slice)
                    axes[current_col].imshow(mri_slice, cmap='gray')
                    axes[current_col].imshow(pred_mask, cmap=brats_cmap, vmin=0, vmax=3, alpha=0.6)
                    axes[current_col].set_title(f"Pred: {model_name}", fontsize=12, pad=10)
                    axes[current_col].axis('off')
                    current_col += 1
                
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown("Legend: Red: NCR/NET | Green: Edema | Blue: Enhancing Tumor")
                
                st.markdown("### Model Performance Comparison")
                m_cols = st.columns(len(active_models))
                for idx, m_name in enumerate(active_models):
                    with m_cols[idx]:
                        st.markdown(f"**{m_name}**")
                        metrics = MODEL_METRICS.get(m_name, {})
                        st.write(f"DSC: {metrics.get('DSC', 'N/A')}")
                        st.write(f"HD: {metrics.get('HD', 'N/A')} mm")
            else:
                st.info("Upload your scan and select models to begin comparison.")

    elif page == "2. Model Information":
        st.title("Model Selection Scoreboard")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("Best Volumetric Overlap\n\n### UNet++\n**DSC: 0.1661**\n\nScenario: Volume priority.")
        with col2:
            st.success("Best Boundary Precision\n\n### SegResNet\n**HD: 8.51 mm**\n\nScenario: Surgical precision.")
        with col3:
            st.warning("Best Overall Balance\n\n### SegResNet\n**S_comp: 0.2600**\n\nScenario: Reliable all-rounder.")
        
        st.markdown("---")
        st.table(dict(sorted(MODEL_METRICS.items(), key=lambda item: float(item[1]['DSC']), reverse=True)))

if __name__ == "__main__":
    main()