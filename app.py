import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Define the models mapped to their weight files
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

# Pre-extracted optimal metrics from the experimental summary
MODEL_METRICS = {
    "SegResNet": {"DSC": "0.1533", "IoU": "0.1068", "HD": "8.5154", "S_composite": "0.2600"},
    "Attention U-Net": {"DSC": "0.1601", "IoU": "0.1123", "HD": "11.2444", "S_composite": "0.2437"},
    "UNet++": {"DSC": "0.1661", "IoU": "0.1221", "HD": "12.4439", "S_composite": "0.2411"},
    "CMAE": {"DSC": "0.1321", "IoU": "0.0938", "HD": "15.7839", "S_composite": "0.2012"},
    "Swin UNETR": {"DSC": "0.1582", "IoU": "0.1113", "HD": "20.2751", "S_composite": "0.2005"},
    "BYOL": {"DSC": "0.1053", "IoU": "0.0680", "HD": "13.7708", "S_composite": "0.1925"},
    "MAE-B": {"DSC": "0.1403", "IoU": "0.0902", "HD": "18.9988", "S_composite": "0.1916"},
    "MAE": {"DSC": "0.1328", "IoU": "0.0876", "HD": "18.3524", "S_composite": "0.1897"},
    "DINO": {"DSC": "0.1021", "IoU": "0.0653", "HD": "18.4960", "S_composite": "0.1694"}
}

@st.cache_data
def load_nifti_file(file_buffer):
    """Loads a NIfTI file and returns the 3D numpy array, rotated for correct anatomical viewing."""
    with open("temp.nii", "wb") as f:
        f.write(file_buffer.getbuffer())
    img = nib.load("temp.nii")
    data = img.get_fdata()
    # BraTS dataset images often need a 90-degree rotation to face upright in Matplotlib
    data = np.rot90(data, k=1, axes=(0, 1))
    return data

def run_model_inference_3d(image_volume, model_name):
    """Placeholder 3D inference function returning a volume of the same shape."""
    mask_volume = np.zeros_like(image_volume)
    # Creating a mock 3D tumor in the center of the volume for demonstration
    cx, cy, cz = image_volume.shape[0]//2, image_volume.shape[1]//2, image_volume.shape[2]//2
    mask_volume[cx-30:cx+30, cy-40:cy+20, cz-20:cz+20] = 1 
    return mask_volume

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
                    # Store data in session state so slider changes don't trigger reloading
                    with st.spinner(f"Processing 3D volume with {selected_model}..."):
                        st.session_state['mri_data'] = load_nifti_file(uploaded_mri)
                        st.session_state['pred_mask'] = run_model_inference_3d(st.session_state['mri_data'], selected_model)
                        
                        if uploaded_gt is not None:
                            st.session_state['gt_data'] = load_nifti_file(uploaded_gt)
                        else:
                            st.session_state['gt_data'] = None
                            
                    st.session_state['run'] = True
                else:
                    st.error("Please upload an MRI file first.")
                    st.session_state['run'] = False
                
        with col2:
            st.subheader("Interactive 3D Dashboard")
            if 'run' in st.session_state and st.session_state['run']:
                mri_vol = st.session_state['mri_data']
                pred_vol = st.session_state['pred_mask']
                gt_vol = st.session_state['gt_data']
                
                # Interactive slider for the Z-axis
                max_slice = mri_vol.shape[2] - 1
                slice_idx = st.slider("Navigate Brain Slices (Z-axis)", 0, max_slice, max_slice // 2)
                
                # Extract 2D slices based on slider position
                mri_slice = mri_vol[:, :, slice_idx]
                pred_slice = pred_vol[:, :, slice_idx]
                
                num_plots = 3 if gt_vol is None else 4
                fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
                
                # 1. Original MRI
                axes[0].imshow(mri_slice, cmap='gray')
                axes[0].set_title(f"Input MRI (Slice {slice_idx})", fontsize=14, pad=10)
                axes[0].axis('off')
                
                # 2. Predicted Mask Overlay
                axes[1].imshow(mri_slice, cmap='gray')
                axes[1].imshow(pred_slice, cmap='Reds', alpha=0.5)
                axes[1].set_title("Predicted Overlay", fontsize=14, pad=10)
                axes[1].axis('off')
                
                # 3. Ground Truth Overlay (If uploaded)
                if gt_vol is not None:
                    gt_slice = gt_vol[:, :, slice_idx]
                    axes[2].imshow(mri_slice, cmap='gray')
                    axes[2].imshow(gt_slice, cmap='Greens', alpha=0.5)
                    axes[2].set_title("Ground Truth Overlay", fontsize=14, pad=10)
                    axes[2].axis('off')
                    
                    # 4. Isolated Prediction Comparison
                    axes[3].imshow(pred_slice, cmap='Reds', alpha=0.5, label='Prediction')
                    axes[3].imshow(gt_slice, cmap='Greens', alpha=0.3, label='Ground Truth')
                    axes[3].set_title("Pred vs Truth", fontsize=14, pad=10)
                    axes[3].axis('off')
                else:
                    # Isolated Prediction (No GT)
                    axes[2].imshow(pred_slice, cmap='viridis')
                    axes[2].set_title("Predicted Mask Isolated", fontsize=14, pad=10)
                    axes[2].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                model_dsc = MODEL_METRICS.get(selected_model, {}).get("DSC", "N/A")
                model_hd = MODEL_METRICS.get(selected_model, {}).get("HD", "N/A")
                    
                st.markdown("### Expected Clinical Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric(label="Expected DSC", value=model_dsc)
                m2.metric(label="Expected HD", value=f"{model_hd} mm")
                m3.metric(label="S_composite", value=MODEL_METRICS.get(selected_model, {}).get("S_composite", "N/A"))
            else:
                st.info("👈 Upload your T1/T2 scan, an optional segmentation mask, and click 'Run Analysis'.")

    elif page == "2. Model Information":
        st.title("Model Architectures & Metrics")
        st.markdown("Detailed performance breakdown for the evaluated segmentation models.")
        st.table(MODEL_METRICS)

if __name__ == "__main__":
    main()