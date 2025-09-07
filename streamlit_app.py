import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import torch
import numpy as np
import tempfile
import os
import plotly.graph_objects as go

# Disable Open3D to prevent segmentation faults on Streamlit Cloud
OPEN3D_AVAILABLE = False

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
    if not OPEN3D_AVAILABLE:
        return None, None
    
    try:
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
        
        # Simplified processing to avoid segfault
        if len(pcd_raw.points) == 0:
            return None, None
            
        # Basic outlier removal (reduced parameters)
        cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        pcd = pcd_raw.select_by_index(ind)
        
        if len(pcd.points) == 0:
            return None, None
        
        # Estimate normals
        pcd.estimate_normals()
        
        # Simplified Poisson reconstruction (reduced depth)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]
        
        if len(mesh.vertices) == 0:
            return None, None
            
        rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        mesh.rotate(rotation, center=(0, 0, 0))
        
        return mesh, pcd
        
    except Exception as e:
        st.error(f"Open3D processing failed: {str(e)}")
        return None, None

def create_3d_point_cloud_fallback(image, depth_output):
    width, height = image.size
    image_array = np.array(image)
    
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Camera parameters
    focal_length = 500
    cx, cy = width / 2, height / 2
    
    # Convert to 3D coordinates
    z = depth_output
    x_3d = (x - cx) * z / focal_length
    y_3d = (y - cy) * z / focal_length
    
    # Flatten and subsample
    step = 4
    x_flat = x_3d.flatten()[::step]
    y_flat = y_3d.flatten()[::step]
    z_flat = z.flatten()[::step]
    colors = image_array.reshape(-1, 3)[::step] / 255.0
    
    return x_flat, y_flat, z_flat, colors

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

def point_cloud_to_plotly(x, y, z, colors):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=colors,
                opacity=0.8
            )
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
        title="3D Point Cloud"
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
        st.image(image, width='stretch')
    
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
        
        # Create 3D point cloud visualization
        x, y, z, colors = create_3d_point_cloud_fallback(processed_image, depth_output)
        plotly_fig = point_cloud_to_plotly(x, y, z, colors)
        st.subheader("3D Point Cloud Reconstruction")
            
        st.plotly_chart(plotly_fig, width='stretch')
        
        # Download options
        st.subheader("Download 3D Data")
        if st.button("Save Point Cloud as CSV"):
            try:
                point_data = np.column_stack([x, y, z, colors])
                csv_data = "x,y,z,r,g,b\n"
                for point in point_data:
                    csv_data += f"{point[0]:.6f},{point[1]:.6f},{point[2]:.6f},{point[3]:.6f},{point[4]:.6f},{point[5]:.6f}\n"
                
                st.download_button(
                    label="Download CSV file",
                    data=csv_data,
                    file_name="3d_point_cloud.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Failed to create CSV: {str(e)}")

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
        st.write("**Output: 3D Point Cloud**") 
        st.write("🎯 Interactive 3D point cloud")

# Processing time and model info
st.markdown("---")
st.info("⏱️ **Note:** 3D model generation typically takes 2-3 minutes depending on image size and system performance.")
st.warning("📝 **Disclaimer:** This uses a simplified depth estimation model for demonstration purposes. Results may not match advanced commercial 3D reconstruction systems.")
