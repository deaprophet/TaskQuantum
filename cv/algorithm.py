"""
Classical feature matching using SIFT + RANSAC.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

class SIFTMatcher:
    """SIFT-based image matching."""
    
    def __init__(self, ratio_threshold=0.8, ransac_threshold=10.0, min_matches=4, nfeatures=10000):
        """
        Initialize matcher.
        
        Args:
            ratio_threshold: Lowe's ratio test threshold (higher = more matches)
            ransac_threshold: RANSAC reprojection threshold in pixels
            min_matches: Minimum number of matches required
            nfeatures: Maximum number of SIFT features to detect
        """
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches
        
        #Initialize SIFT detector
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            contrastThreshold=0.03,
            edgeThreshold=15
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def preprocess_for_matching(self, image):
        """
        Preprocess image for better feature matching.
        
        Args:
            image: RGB or grayscale image
        
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Histogram equalization
        gray = cv2.equalizeHist(gray)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        return gray
    
    def detect_and_compute(self, image):
        """
        Detect keypoints and compute descriptors.
        
        Args:
            image: RGB or grayscale image
        
        Returns:
            (keypoints, descriptors)
        """
        # Use preprocessing
        gray = self.preprocess_for_matching(image)
        
        kp, des = self.sift.detectAndCompute(gray, None)
        return kp, des
    
    def match_features(self, des1, des2):
        """
        Match descriptors with ratio test.
        
        Args:
            des1: Descriptors from image 1
            des2: Descriptors from image 2
        
        Returns:
            List of good matches
        """
        # KNN match with k=2
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def filter_with_ransac(self, kp1, kp2, matches):
        """
        Filter matches using RANSAC.
        
        Args:
            kp1: Keypoints from image 1
            kp2: Keypoints from image 2
            matches: List of matches
        
        Returns:
            (points1, points2, inlier_mask)
        """
        if len(matches) < self.min_matches:
            return np.array([]), np.array([]), np.array([])
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # RANSAC
        H, mask = cv2.findHomography(
            pts1, pts2,
            cv2.RANSAC,
            self.ransac_threshold
        )
        
        if mask is None:
            return np.array([]), np.array([]), np.array([])
        
        mask = mask.ravel().astype(bool)
        
        return pts1[mask], pts2[mask], mask
    
    def match_image_pair(self, img1, img2):
        """
        Complete matching pipeline for an image pair.
        
        Args:
            img1: First image (RGB)
            img2: Second image (RGB)
        
        Returns:
            Dictionary with matching results
        """
        # Detect and compute
        kp1, des1 = self.detect_and_compute(img1)
        kp2, des2 = self.detect_and_compute(img2)
        
        # Match
        matches = self.match_features(des1, des2)
        
        # Filter with RANSAC
        pts1, pts2, mask = self.filter_with_ransac(kp1, kp2, matches)
        
        return {
            'keypoints1': kp1,
            'keypoints2': kp2,
            'descriptors1': des1,
            'descriptors2': des2,
            'raw_matches': matches,
            'num_raw_matches': len(matches),
            'inlier_points1': pts1,
            'inlier_points2': pts2,
            'num_inliers': len(pts1),
            'inlier_ratio': len(pts1) / len(matches) if len(matches) > 0 else 0
        }


def load_image_pair(pair_dir):
    """Load image pair from directory."""
    pair_path = Path(pair_dir)
    img1 = np.array(Image.open(pair_path / 'img1.png'))
    img2 = np.array(Image.open(pair_path / 'img2.png'))
    return img1, img2

