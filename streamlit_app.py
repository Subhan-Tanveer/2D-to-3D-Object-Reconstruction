import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import torch
import numpy as np
import open3d as o3d
import tempfile
import os
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="2D to 3D Reconstruction", layout="wide")

# Cache model loading
@st.cache_resource
def load_model():
    torch.manual_seed(42)
    np.random.seed(42)
    feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
    return feature_extractor, model

def process_image(image, feature_extractor, model):
    # Resize image
    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width, new_height)
    image = image.resize(new_size)
    
    # Process with model
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.inference_mode():
        output = model(**inputs)
        predicted_depth = output.predicted_depth
    
    # Post-process
    pad = 16
    depth_output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    depth_output = depth_output[pad:-pad, pad:-pad]
    image = image.crop((pad, pad, image.width - pad, image.height - pad))
    
    return image, depth_output

def create_3d_mesh(image, depth_output):
    width, height = image.size
    depth_image = (depth_output * 255 / np.max(depth_output)).astype('uint8')
    image_array = np.array(image)
    
    # Create Open3D objects
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image_array)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_o3d, depth_o3d, convert_rgb_to_intensity=False
    )
    
    # Camera intrinsics
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)
    
    # Create point cloud
    pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    
    # Outlier removal
    cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=30, std_ratio=3.0)
    pcd = pcd_raw.select_by_index(ind)
    
    # Estimate normals
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=15, n_threads=1)[0]
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))
    
    return mesh, pcd

def mesh_to_plotly(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    
    if len(colors) == 0:
        colors = np.ones((len(vertices), 3)) * 0.7
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1], 
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            vertexcolor=colors,
            opacity=0.8
        )
    ])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=800,
        height=600,
        title="3D Reconstruction"
    )
    
    return fig

# Main app
st.title("🎯 2D to 3D Object Reconstruction")
st.write("Upload a 2D image to generate its 3D reconstruction using depth estimation")

# Load model
with st.spinner("Loading AI model..."):
    feature_extractor, model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    # Process image
    with st.spinner("Processing image and generating 3D model..."):
        processed_image, depth_output = process_image(image, feature_extractor, model)
        
        # Show depth map
        with col2:
            st.subheader("Depth Map")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(depth_output, cmap='plasma')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
        
        # Create 3D mesh
        mesh, pcd = create_3d_mesh(processed_image, depth_output)
        
        # Convert to plotly and display
        plotly_fig = mesh_to_plotly(mesh)
        st.subheader("3D Reconstruction")
        st.plotly_chart(plotly_fig, use_container_width=True)
        
        # Download options
        st.subheader("Download 3D Model")
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("Save as PLY"):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
                    o3d.io.write_triangle_mesh(tmp_file.name, mesh)
                    with open(tmp_file.name, 'rb') as f:
                        st.download_button(
                            label="Download PLY file",
                            data=f.read(),
                            file_name="3d_model.ply",
                            mime="application/octet-stream"
                        )
                    os.unlink(tmp_file.name)
        
        with col4:
            if st.button("Save Point Cloud"):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pcd') as tmp_file:
                    o3d.io.write_point_cloud(tmp_file.name, pcd)
                    with open(tmp_file.name, 'rb') as f:
                        st.download_button(
                            label="Download PCD file", 
                            data=f.read(),
                            file_name="point_cloud.pcd",
                            mime="application/octet-stream"
                        )
                    os.unlink(tmp_file.name)

else:
    st.info("Please upload an image to start the 3D reconstruction process")
    
    # Show example
    st.subheader("Example")
    st.write("Here's what the app can do with your images:")
    example_col1, example_col2 = st.columns(2)
    with example_col1:
        st.write("**Input: 2D Image**")
        st.write("📷 Upload any photo")
    with example_col2:
        st.write("**Output: 3D Model**") 
        st.write("🎯 Interactive 3D reconstruction")

# Processing time and model info
st.markdown("---")
st.info("⏱️ **Note:** 3D model generation typically takes 2-3 minutes depending on image size and system performance.")
st.warning("📝 **Disclaimer:** This uses a simplified depth estimation model for demonstration purposes. Results may not match advanced commercial 3D reconstruction systems.")