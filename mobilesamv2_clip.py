import os
import argparse
import time
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from typing import Tuple, Dict, Any, Generator, List, Type
from dataclasses import dataclass, field

# Add MobileSAMv2 to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "submodules/MobileSAM/MobileSAMv2"))

from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import sam_model_registry, SamPredictor
from utils.openclip_encoder import OpenCLIPNetwork

# Model weight paths
ENCODER_PATHS = {
    'efficientvit_l2': f'{current_dir}/submodules/MobileSAM/MobileSAMv2/weight/l2.pt',
    'tiny_vit': f'{current_dir}/submodules/MobileSAM/MobileSAMv2/weightmobile_sam.pt',
    'sam_vit_h': './ckpts/sam_vit_h_4b8939.pth',
}

class MobileSAM2_CLIP:
    """MobileSAMv2 with CLIP integration for image segmentation and feature extraction."""
    
    def __init__(self, args: argparse.Namespace, device: str = "cuda"):
        """Initialize MobileSAM2_CLIP.
        
        Args:
            args: Parsed command line arguments
            device: Computation device ('cuda' or 'cpu')
        """
        self.device = device
        self.args = args
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize models
        mobilesamv2, ObjAwareModel = self._create_model()
        image_encoder = sam_model_registry[args.encoder_type](ENCODER_PATHS[args.encoder_type])
        mobilesamv2.image_encoder = image_encoder
        
        mobilesamv2.to(device=device)
        mobilesamv2.eval()
        predictor = SamPredictor(mobilesamv2)
        
        self.mobilesamv2 = mobilesamv2
        self.object_aware_model = ObjAwareModel
        self.predictor = predictor
        self.clip_model = OpenCLIPNetwork(device="cuda")
        self.resize_transform = Resize((args.resize_size, args.resize_size))
        
        # Initialize random colors for visualization
        np.random.seed(0)
        self.label2colors = np.random.randint(0, 256, size=(500, 3))
        self.label2colors[0] = 0
        
        self.counts = 0
        self.total_time = 0

    def _create_model(self) -> Tuple[nn.Module, ObjectAwareModel]:
        """Initialize MobileSAMv2 model components.
        
        Returns:
            Tuple: (MobileSAMv2 model, ObjectAwareModel)
        """
        Prompt_guided_path = self.args.Prompt_guided_Mask_Decoder_path
        obj_model_path = self.args.ObjectAwareModel_path
        
        ObjAwareModel = ObjectAwareModel(obj_model_path)
        PromptGuidedDecoder = sam_model_registry['PromptGuidedDecoder'](Prompt_guided_path)
        
        mobilesamv2 = sam_model_registry['vit_h']()
        mobilesamv2.prompt_encoder = PromptGuidedDecoder['PromtEncoder']
        mobilesamv2.mask_decoder = PromptGuidedDecoder['MaskDecoder']
        
        return mobilesamv2, ObjAwareModel

    def parse_one_image(self, image: np.ndarray, image_name: str) -> Tuple[np.ndarray, torch.Tensor]:
        """Process a single image and generate segmentation map and CLIP embeddings.
        
        Args:
            image: Input image as numpy array
            image_name: Name of the image file
            
        Returns:
            Tuple: (label_map, clip_embeddings)
        """
        start_time = time.time()
        
        # Object detection and segmentation
        obj_results = self.object_aware_model(
            image, device=self.device, retina_masks=self.args.retina,
            imgsz=self.args.imgsz, conf=self.args.conf, iou=self.args.iou
        )
        
        self.predictor.set_image(image)
        input_boxes = obj_results[0].boxes.xyxy.cpu().numpy()
        input_boxes = self.predictor.transform.apply_boxes(input_boxes, self.predictor.original_size)
        input_boxes = torch.from_numpy(input_boxes).cuda()

        # Generate embeddings and masks
        image_embedding = self.predictor.features
        image_embedding = torch.repeat_interleave(image_embedding, 320, dim=0)
        prompt_embedding = self.mobilesamv2.prompt_encoder.get_dense_pe()
        prompt_embedding = torch.repeat_interleave(prompt_embedding, 320, dim=0)

        sam_mask, worked_bbox, sam_mask_confs = [], [], []
        for (boxes,) in self._batch_iterator(320, input_boxes):
            with torch.no_grad():
                image_embedding = image_embedding[0:boxes.shape[0],:,:,:]
                prompt_embedding = prompt_embedding[0:boxes.shape[0],:,:,:]
                
                sparse_embeddings, dense_embeddings = self.mobilesamv2.prompt_encoder(
                    points=None, boxes=boxes, masks=None
                )
                
                low_res_masks, conf = self.mobilesamv2.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=prompt_embedding,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    simple_type=True,
                )
                
                low_res_masks = self.predictor.model.postprocess_masks(
                    low_res_masks, self.predictor.input_size, self.predictor.original_size
                )
                sam_mask_pre = (low_res_masks > self.mobilesamv2.mask_threshold)*1.0
                sam_mask.append(sam_mask_pre.squeeze(1))
                worked_bbox.append(boxes)
                sam_mask_confs.append(conf)

        # Combine results
        sam_mask = torch.cat(sam_mask)
        sam_mask_confs = torch.cat(sam_mask_confs)
        worked_bbox = torch.cat(worked_bbox)
        
        assert len(sam_mask) == len(sam_mask_confs) == len(worked_bbox), (
            "Masks, confidences and boxes should have the same length"
        )

        # Generate label map and process segmented images
        label_map, conf_values, seg_imgs = self._generate_label_map(
            image, sam_mask, worked_bbox, sam_mask_confs
        )

        # Extract CLIP embeddings
        with torch.no_grad():
            clip_embeddings = self.clip_model.encode_image(seg_imgs)
            clip_embeddings /= clip_embeddings.norm(dim=-1, keepdim=True)
            clip_embeddings = clip_embeddings.half()
            clip_embeddings = torch.cat([
                torch.zeros(1, clip_embeddings.size(1), device=self.device, dtype=clip_embeddings.dtype), 
                clip_embeddings
            ], dim=0)
            
        # Performance tracking
        self.total_time += time.time() - start_time
        self.counts += 1
        print(f"FPS: {self.counts / self.total_time}")
            
        if self.args.save_results:
            self._save_results(label_map, clip_embeddings, image_name)
        
        return label_map.cpu().numpy(), clip_embeddings.cpu()

    def _save_results(self, label_map: torch.Tensor, clip_embeddings: torch.Tensor, image_name: str):
        """Save segmentation results and embeddings to disk.
        
        Args:
            label_map: Segmentation label map
            clip_embeddings: CLIP feature embeddings
            image_name: Name of the input image
        """
        cpu_label_map = label_map.cpu().numpy()
        rgb_img = self.label2colors[cpu_label_map]
        
        os.makedirs(os.path.join(self.args.output_dir, "mobile_sam"), exist_ok=True)
        img_path = os.path.join(self.args.output_dir, "mobile_sam", image_name)
        cv2.imwrite(img_path, rgb_img)
        
        # Save label map and embeddings
        np.save(os.path.join(self.args.output_dir, image_name[:-4]+"_s.npy"), cpu_label_map[None, ...])
        saved_clip_embed = clip_embeddings.cpu().numpy()
        np.save(os.path.join(self.args.output_dir, image_name[:-4]+"_f.npy"), saved_clip_embed)
        
        assert cpu_label_map.max() == len(saved_clip_embed) - 1, (
            "Label map max value should match embedding length"
        )

    @staticmethod
    def _batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
        """Batch iterator for processing inputs in chunks.
        
        Args:
            batch_size: Size of each batch
            *args: Input tensors to batch
            
        Yields:
            List of batched tensors
        """
        assert len(args) > 0 and all(
            len(a) == len(args[0]) for a in args
        ), "All inputs must have the same size for batching"
        
        n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
        for b in range(n_batches):
            yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

    def _generate_label_map(self, 
                          image: np.ndarray, 
                          masks: torch.Tensor,
                          boxes: torch.Tensor, 
                          confs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate label map and process segmented images.
        
        Args:
            image: Original input image
            masks: Segmentation masks
            boxes: Bounding boxes
            confs: Confidence scores
            
        Returns:
            Tuple: (label_map, confidence_values, segmented_images)
        """
        label_map = torch.zeros((image.shape[0], image.shape[1]), dtype=torch.int32, device=self.device)
        seg_img_list = []
        conf_values = torch.zeros(masks.size(0) + 1, device=self.device)
        group_counter = 1
        
        image = torch.from_numpy(image).float().to('cuda')

        for i in range(masks.size(0)):
            mask = masks[i].bool()
            box = boxes[i]

            seg_img = self._gpu_get_seg_img(mask, box, image)
            padded_seg_img = self._pad_img(seg_img)
            resized_seg_img = self.resize_transform(padded_seg_img.permute(2, 0, 1)).permute(1, 2, 0)
            
            seg_img_list.append(resized_seg_img)
            label_map[mask] = group_counter
            conf_values[group_counter] = confs[i].item()
            group_counter += 1

        seg_imgs = torch.stack(seg_img_list).permute(0, 3, 1, 2) / 255.0
        return label_map, conf_values, seg_imgs

    @staticmethod
    def _gpu_get_seg_img(mask: torch.Tensor, box: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Extract segmented region from image using mask.
        
        Args:
            mask: Segmentation mask
            box: Bounding box coordinates
            image: Original image
            
        Returns:
            Segmented image region
        """
        region = image.clone()
        region[~mask] = 0  # Mask out background
        
        x, y, w, h = MobileSAM2_CLIP._find_counter_opencv(mask.cpu().numpy())
        return region[y:y + h, x:x + w, ...]

    @staticmethod
    def _find_counter_opencv(mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Find bounding rectangle using OpenCV.
        
        Args:
            mask: Binary mask
            
        Returns:
            Tuple: (x, y, width, height) of bounding rectangle
        """
        counters, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cv2.boundingRect(counters[0])

    @staticmethod
    def _pad_img(img: torch.Tensor) -> torch.Tensor:
        """Pad image to make it square.
        
        Args:
            img: Input image tensor
            
        Returns:
            Padded square image
        """
        h, w, _ = img.shape
        l = max(h, w)
        padded = torch.zeros((l, l, 3), dtype=torch.uint8, device=img.device)

        if h > w:
            padded[:, (h - w) // 2:(h - w) // 2 + w, :] = img
        else:
            padded[(w - h) // 2:(w - h) // 2 + h, :, :] = img

        return padded

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for MobileSAMv2 configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="MobileSAMv2 with CLIP integration")
    
    # Model paths
    parser.add_argument("--ObjectAwareModel_path", type=str, 
                       default=f'{current_dir}/submodules/MobileSAM/MobileSAMv2/weight/ObjectAwareModel.pt', 
                       help="Path to ObjectAwareModel weights")
    parser.add_argument("--Prompt_guided_Mask_Decoder_path", type=str, 
                       default=f'{current_dir}/submodules/MobileSAM/MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt', 
                       help="Path to Prompt Guided Mask Decoder weights")
    parser.add_argument("--encoder_path", type=str, 
                       default=f"{current_dir}/submodules/MobileSAM/MobileSAMv2/weights/mobile_sam.pt", 
                       help="Custom encoder weights path")
    
    # Data parameters
    parser.add_argument("--image_folder", type=str, required=True,
                       help="Path to input images folder")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    
    # Processing parameters
    parser.add_argument("--imgsz", type=int, default=480, help="Input image size")
    parser.add_argument("--resize_size", type=int, default=224, 
                       help="Size for resizing segmented images")
    parser.add_argument("--iou", type=float, default=0.9, help="YOLO IoU threshold")
    parser.add_argument("--conf", type=float, default=0.4, 
                       help="YOLO confidence threshold")
    
    # Model selection
    parser.add_argument("--encoder_type", 
                       choices=['tiny_vit', 'sam_vit_h', 'mobile_sam', 
                               'efficientvit_l2', 'efficientvit_l1', 'efficientvit_l0'], 
                       default='efficientvit_l2', 
                       help="Model encoder type")
    
    # Runtime options
    parser.add_argument("--retina", action="store_true", 
                       help="Draw high-quality segmentation masks")
    parser.add_argument("--save_results", action="store_true",
                       help="Save output segmentation maps and embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Computation device to use")
    
    return parser.parse_args()

def main():
    """Main execution function for MobileSAMv2 processing."""
    args = parse_args()
    
    # Initialize the processor
    processor = MobileSAM2_CLIP(args, device=args.device)
    
    # Process all images in the folder
    image_names = sorted(os.listdir(args.image_folder))
    if not image_names:
        print("No images found in the specified folder.")
        return
    
    for image_name in image_names:
        image_path = os.path.join(args.image_folder, image_name)
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image {image_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label_map, clip_embeddings = processor.parse_one_image(image, image_name)
            
            print(f"Processed {image_name}:")
            print(f"  Label map shape: {label_map.shape}")
            print(f"  CLIP embeddings shape: {clip_embeddings.shape}")
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
    
    '''  run example:
    python mobilesamv2_clip.py \
    --image_folder /path/to/images  \
    --output_dir /path/to/output \
    --save_results 
    '''
    