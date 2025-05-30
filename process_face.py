import os
import json
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import face_alignment
import cv2
import pickle
import sys

# Add MODNet to path
sys.path.append('MODNet/src')
from models.modnet import MODNet

class FaceProcessor:
    def __init__(self, device='cuda'):
        self.device = device
        self.setup_modnet()
        self.setup_flame()
        
    def setup_modnet(self):
        # Load MODNet model
        self.modnet = MODNet(backbone_pretrained=False)
        
        # Load pretrained weights
        modnet_ckpt_path = 'pretrained/MODNet/modnet_photographic_portrait_matting.ckpt'
        self.modnet = torch.nn.DataParallel(self.modnet)
        self.modnet.load_state_dict(torch.load(modnet_ckpt_path, map_location=self.device))
        self.modnet.to(self.device)
        self.modnet.eval()
        
        # MODNet preprocessing transform (correct normalization)
        self.modnet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
    def generate_modnet_mask(self, image_path):
        """Generate mask using MODNet model"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to multiple of 32 for MODNet
        ref_size = 512
        im = np.asarray(image)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # Resize
        im = cv2.resize(im, (ref_size, ref_size), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL and apply transform
        im_pil = Image.fromarray(im)
        input_tensor = self.modnet_transform(im_pil).unsqueeze(0).to(self.device)
        
        # Generate mask with MODNet
        with torch.no_grad():
            _, _, matte = self.modnet(input_tensor, True)
            
        # Process the output - keep as continuous values, don't binarize
        matte = F.interpolate(matte, size=(ref_size, ref_size), mode='bilinear', align_corners=False)
        matte = matte.squeeze().cpu().numpy()
        
        # Apply some post-processing to sharpen the mask
        # Clip values
        matte = np.clip(matte, 0, 1)
        
        # Apply power function to sharpen edges
        matte = np.power(matte, 1.2)
        
        # Convert to 8-bit
        mask = (matte * 255).astype(np.uint8)
        
        return mask
        
    def setup_flame(self):
        # Load FLAME model
        flame_model_path = 'pretrained/FLAME/generic_model.pkl'
        with open(flame_model_path, 'rb') as f:
            self.flame_model = pickle.load(f, encoding='latin1')
        
        # FLAME parameters
        self.n_shape = 100  # Shape parameters
        self.n_exp = 50     # Expression parameters
        self.n_pose = 15    # Pose parameters (including jaw, neck, etc.)
        
        # Get FLAME vertices and faces
        self.faces = self.flame_model['f']
        self.shapedirs = self.flame_model['shapedirs']  # Shape blend shapes
        self.posedirs = self.flame_model['posedirs']    # Pose blend shapes
        self.J_regressor = self.flame_model['J_regressor']  # Joint regressor
        self.parents = self.flame_model['kintree_table'][0]  # Kinematic tree
        self.v_template = self.flame_model['v_template']    # Template vertices
        self.weights = self.flame_model['weights']          # Skinning weights
        
        # Expression blend shapes
        if 'expressiondir' in self.flame_model:
            self.expressiondir = self.flame_model['expressiondir']
        else:
            # Fallback if expression blend shapes are not available
            self.expressiondir = np.zeros((self.v_template.shape[0], 3, self.n_exp))
        
    def process_image(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512))  # Resize to standard size
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
        
        # Generate mask using MODNet
        mask_np = self.generate_modnet_mask(image_path)
        
        # Convert mask to tensor
        mask = torch.from_numpy(mask_np).float() / 255.0
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Save mask for debugging
        mask_dir = os.path.join(os.path.dirname(image_path).replace('ori_imgs', 'mask'))
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, os.path.basename(image_path).replace('.jpg', '.png').replace('.jpeg', '.png'))
        
        mask_pil = Image.fromarray(mask_np)
        mask_pil.save(mask_path)
        
        # Extract 3DMM parameters
        landmarks = self.detect_landmarks(image_tensor)
        face_params = self.fit_3dmm(landmarks)
        
        return mask, face_params
        
    def detect_landmarks(self, image):
        # Convert tensor to numpy array
        image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # Initialize face alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=self.device)
        
        # Detect landmarks
        landmarks = fa.get_landmarks(image_np)
        if landmarks is None:
            raise ValueError("No face detected in the image")
            
        return landmarks[0]  # Return first face's landmarks
        
    def fit_3dmm(self, landmarks):
        # Initialize FLAME parameters
        shape_params = np.zeros(self.n_shape)
        exp_params = np.zeros(self.n_exp) 
        pose_params = np.zeros(self.n_pose)
        
        # Simplified fitting - in practice you'd use optimization
        # For demo purposes, we'll generate reasonable parameters
        
        # Generate some expression variation based on landmark positions
        # This is a simplified approach - real fitting would use optimization
        if landmarks.shape[0] >= 68:  # Standard 68 landmark format
            # Use mouth landmarks to estimate expression
            mouth_landmarks = landmarks[48:68]  # Mouth region
            mouth_height = mouth_landmarks[:, 1].max() - mouth_landmarks[:, 1].min()
            mouth_width = mouth_landmarks[:, 0].max() - mouth_landmarks[:, 0].min()
            
            # Simple expression estimation
            exp_params[0] = mouth_height * 0.01  # Jaw open
            exp_params[1] = mouth_width * 0.005   # Mouth stretch
            
        # Estimate pose from landmarks
        pose_params = self.estimate_pose_from_landmarks(landmarks)
        
        # Generate FLAME mesh
        vertices = self.generate_flame_mesh(shape_params, exp_params, pose_params)
        
        # Project to get 2D landmarks
        projected_landmarks = self.project_flame_landmarks(vertices)
        
        # Estimate camera parameters
        camera_params = self.estimate_camera_flame(landmarks, projected_landmarks)
        
        # Generate transform matrix
        transform_matrix = self.generate_transform_matrix(camera_params)
        
        # Get face rectangle
        face_rect = self.get_face_rect(landmarks)
        
        return {
            "transform": transform_matrix,
            "exp": exp_params[:21],  # First 21 expression parameters for compatibility
            "exp_ori": np.concatenate([shape_params[:20], exp_params, pose_params[:6]]),  # Combined parameters
            "face_rect": face_rect
        }
        
    def generate_flame_mesh(self, shape_params, exp_params, pose_params):
        """Generate FLAME mesh from parameters"""
        # Start with template - handle chumpy object
        if hasattr(self.v_template, 'r'):
            vertices = self.v_template.r.copy()
        else:
            vertices = self.v_template.copy()
        
        # Add shape variation
        shapedirs = self.shapedirs.r if hasattr(self.shapedirs, 'r') else self.shapedirs
        for i in range(min(len(shape_params), shapedirs.shape[-1])):
            vertices += shape_params[i] * shapedirs[:, :, i]
            
        # Add expression variation
        expressiondir = self.expressiondir.r if hasattr(self.expressiondir, 'r') else self.expressiondir
        for i in range(min(len(exp_params), expressiondir.shape[-1])):
            vertices += exp_params[i] * expressiondir[:, :, i]
            
        return vertices
        
    def project_flame_landmarks(self, vertices):
        """Project FLAME vertices to get landmark positions"""
        # FLAME landmark indices (simplified - you'd want the actual landmark indices)
        landmark_indices = np.array([
            # Approximate landmark indices for FLAME
            # This is simplified - real implementation would use proper landmark mappings
            2212, 3061, 3485, 3384, 3386, 3389, 3393, 3398, 3407,  # Jaw line
            3064, 3065, 3066, 3067, 3068,  # Right eyebrow
            2966, 2967, 2968, 2969, 2970,  # Left eyebrow
            3051, 3052, 3053, 3054,  # Nose bridge
            3055, 3056, 3057, 3058, 3059,  # Nose tip
            # Add more landmark indices as needed
        ])
        
        # Ensure we don't exceed vertex count
        landmark_indices = landmark_indices[landmark_indices < len(vertices)]
        
        if len(landmark_indices) < 68:
            # Fill with random vertices if we don't have enough landmarks
            additional_indices = np.random.choice(len(vertices), 68 - len(landmark_indices), replace=False)
            landmark_indices = np.concatenate([landmark_indices, additional_indices])
        
        return vertices[landmark_indices[:68]]
        
    def estimate_pose_from_landmarks(self, landmarks):
        """Estimate pose parameters from 2D landmarks"""
        pose_params = np.zeros(self.n_pose)
        
        if landmarks.shape[0] >= 68:
            # Estimate head rotation from face orientation
            left_eye = landmarks[36:42].mean(axis=0)
            right_eye = landmarks[42:48].mean(axis=0)
            nose_tip = landmarks[30]
            
            # Calculate approximate head pose
            eye_center = (left_eye + right_eye) / 2
            eye_direction = right_eye - left_eye
            
            # Simple pose estimation
            pose_params[0] = np.arctan2(eye_direction[1], eye_direction[0]) * 0.1  # Roll
            pose_params[1] = (nose_tip[1] - eye_center[1]) * 0.001  # Pitch
            pose_params[2] = (nose_tip[0] - eye_center[0]) * 0.001  # Yaw
            
        return pose_params
        
    def estimate_camera_flame(self, landmarks_2d, landmarks_3d):
        """Estimate camera parameters for FLAME model"""
        # Ensure we have the same number of points
        min_points = min(len(landmarks_2d), len(landmarks_3d))
        landmarks_2d = landmarks_2d[:min_points]
        landmarks_3d = landmarks_3d[:min_points]
        
        # Camera intrinsics
        camera_matrix = np.array([[600, 0, 256], [0, 600, 256], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros(4, dtype=np.float32)
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            landmarks_3d.astype(np.float32),
            landmarks_2d.astype(np.float32),
            camera_matrix,
            dist_coeffs
        )
        
        if not success:
            # Fallback to identity
            rvec = np.zeros(3, dtype=np.float32)
            tvec = np.array([0, 0, 0.3], dtype=np.float32)
            
        return {
            "rotation": rvec,
            "translation": tvec
        }
        
    def generate_transform_matrix(self, camera_params):
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(camera_params["rotation"])
        
        # Create 4x4 transform matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = camera_params["translation"].reshape(3)
        
        return transform
        
    def get_face_rect(self, landmarks):
        # Get face rectangle from landmarks
        x_min = landmarks[:, 0].min()
        y_min = landmarks[:, 1].min()
        x_max = landmarks[:, 0].max()
        y_max = landmarks[:, 1].max()
        
        return [int(x_min), int(y_min), int(x_max), int(y_max)]
        
    def generate_transforms(self, image_paths, output_path):
        transforms_data = {
            "fx": 600.0,
            "fy": 600.0,
            "cx": 256.0,
            "cy": 256.0,
            "h": 512,
            "w": 512,
            "frames": []
        }
        
        for i, image_path in enumerate(image_paths):
            mask, face_params = self.process_image(image_path)
            
            frame_data = {
                "img_id": i,
                "aud_id": i,
                "near": 0.24,
                "far": 0.48,
                "transform_matrix": face_params["transform"].tolist(),
                "exp": face_params["exp"].tolist(),
                "exp_ori": face_params["exp_ori"].tolist(),
                "face_rect": face_params["face_rect"]
            }
            
            transforms_data["frames"].append(frame_data)
            
        with open(output_path, 'w') as f:
            json.dump(transforms_data, f, indent=2)

if __name__ == "__main__":
    processor = FaceProcessor()
    image_dir = "data/id6/ori_imgs"
    output_path = "data/id6/transforms.json"
    
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    processor.generate_transforms(image_paths, output_path) 