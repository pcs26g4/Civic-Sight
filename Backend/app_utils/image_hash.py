"""
Image Hashing Utilities
Provides functions for calculating and comparing image hashes for deduplication.
Uses perceptual hashing (pHash) to detect similar images even with slight variations.
Updated to support both image bytes and URLs.
"""
import hashlib
from io import BytesIO
from typing import Optional, Union
import requests

try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    Image = None
    imagehash = None


def calculate_perceptual_hash_from_bytes(image_bytes: bytes) -> Optional[str]:
    """
    Calculate perceptual hash (pHash) for an image from bytes.
    This hash is robust to minor variations (compression, resizing, etc.)
    
    Args:
        image_bytes: Image file bytes
        
    Returns:
        Hex string representation of the hash, or None if calculation fails
    """
    if not IMAGEHASH_AVAILABLE:
        # Fallback to MD5 if imagehash is not available
        return calculate_md5_hash(image_bytes)
    
    try:
        img = Image.open(BytesIO(image_bytes))
        # Convert to RGB if necessary (handles RGBA, P, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate perceptual hash (8x8 = 64 bits)
        phash = imagehash.phash(img, hash_size=8)
        return str(phash)
    except Exception as e:
        # Fallback to MD5 on error
        print(f"Perceptual hash calculation failed: {e}")
        return calculate_md5_hash(image_bytes)


def calculate_perceptual_hash_from_url(image_url: str, timeout: int = 10) -> Optional[str]:
    """
    Calculate perceptual hash (pHash) for an image from URL.
    
    Args:
        image_url: URL of the image
        timeout: Request timeout in seconds
        
    Returns:
        Hex string representation of the hash, or None if calculation fails
    """
    if not IMAGEHASH_AVAILABLE:
        # Can't calculate perceptual hash without library
        return None
    
    try:
        # Fetch image from URL
        response = requests.get(image_url, timeout=timeout)
        response.raise_for_status()
        
        # Calculate hash from bytes
        return calculate_perceptual_hash_from_bytes(response.content)
    except Exception as e:
        print(f"Perceptual hash calculation from URL failed: {e}")
        return None


def calculate_md5_hash(image_bytes: bytes) -> str:
    """
    Calculate MD5 hash for exact duplicate detection.
    
    Args:
        image_bytes: Image file bytes
        
    Returns:
        Hex string representation of the MD5 hash
    """
    return hashlib.md5(image_bytes).hexdigest()


def calculate_md5_hash_from_url(image_url: str, timeout: int = 10) -> Optional[str]:
    """
    Calculate MD5 hash for an image from URL.
    
    Args:
        image_url: URL of the image
        timeout: Request timeout in seconds
        
    Returns:
        Hex string representation of the MD5 hash, or None if calculation fails
    """
    try:
        # Fetch image from URL
        response = requests.get(image_url, timeout=timeout)
        response.raise_for_status()
        
        # Calculate MD5 from content
        return hashlib.md5(response.content).hexdigest()
    except Exception as e:
        print(f"MD5 hash calculation from URL failed: {e}")
        return None


def compare_image_hashes(hash1: Optional[str], hash2: Optional[str], threshold: int = 5) -> bool:
    """
    Compare two perceptual hashes to determine if images are similar.
    
    Args:
        hash1: First image hash (hex string)
        hash2: Second image hash (hex string)
        threshold: Maximum Hamming distance to consider images similar (default: 5)
                  Lower values = stricter matching
        
    Returns:
        True if images are similar (within threshold), False otherwise
    """
    if not hash1 or not hash2:
        return False
    
    # If hashes are MD5 (32 chars), do exact comparison
    if len(hash1) == 32 and len(hash2) == 32:
        return hash1 == hash2
    
    # For perceptual hashes, calculate Hamming distance
    try:
        if IMAGEHASH_AVAILABLE:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            distance = h1 - h2  # Hamming distance
            return distance <= threshold
        else:
            # Fallback: exact match for MD5
            return hash1 == hash2
    except Exception as e:
        print(f"Hash comparison failed: {e}")
        # Fallback: exact match
        return hash1 == hash2


def calculate_image_hash(image_bytes: Optional[bytes] = None, 
                        image_url: Optional[str] = None, 
                        use_perceptual: bool = True,
                        timeout: int = 10) -> Optional[str]:
    """
    ✅ UPDATED: Calculate hash for an image from bytes or URL.
    
    Args:
        image_bytes: Image file bytes (optional)
        image_url: URL of the image (optional)
        use_perceptual: If True, use perceptual hash; if False, use MD5
        timeout: Request timeout for URL fetching (seconds)
        
    Returns:
        Hex string representation of the hash, or None if calculation fails
        
    Note:
        At least one of image_bytes or image_url must be provided.
    """
    if image_bytes:
        # Calculate from bytes
        if use_perceptual:
            phash = calculate_perceptual_hash_from_bytes(image_bytes)
            if phash:
                return phash
            # Fallback to MD5 if perceptual hash fails
            return calculate_md5_hash(image_bytes)
        else:
            return calculate_md5_hash(image_bytes)
    
    elif image_url:
        # Calculate from URL
        if use_perceptual:
            phash = calculate_perceptual_hash_from_url(image_url, timeout)
            if phash:
                return phash
            # Fallback to MD5 if perceptual hash fails
            return calculate_md5_hash_from_url(image_url, timeout)
        else:
            return calculate_md5_hash_from_url(image_url, timeout)
    
    else:
        raise ValueError("Either image_bytes or image_url must be provided")


def get_image_hash_for_deduplication(image_bytes: Optional[bytes] = None,
                                   image_url: Optional[str] = None) -> Optional[str]:
    """
    Smart function to get image hash for deduplication.
    Uses perceptual hashing when possible, falls back to MD5.
    
    Args:
        image_bytes: Image file bytes (optional)
        image_url: URL of the image (optional)
        
    Returns:
        Hash string or None if calculation fails
    """
    try:
        return calculate_image_hash(
            image_bytes=image_bytes,
            image_url=image_url,
            use_perceptual=True
        )
    except Exception as e:
        print(f"Failed to calculate deduplication hash: {e}")
        return None


def are_images_similar(image1_bytes: Optional[bytes] = None,
                      image1_url: Optional[str] = None,
                      image2_bytes: Optional[bytes] = None,
                      image2_url: Optional[str] = None,
                      threshold: int = 5) -> bool:
    """
    Compare two images to see if they're similar.
    
    Args:
        image1_bytes/image1_url: First image (provide one)
        image2_bytes/image2_url: Second image (provide one)
        threshold: Hamming distance threshold for perceptual hashes
        
    Returns:
        True if images are similar, False otherwise
    """
    # Calculate hashes for both images
    hash1 = calculate_image_hash(
        image_bytes=image1_bytes,
        image_url=image1_url,
        use_perceptual=True
    )
    
    hash2 = calculate_image_hash(
        image_bytes=image2_bytes,
        image_url=image2_url,
        use_perceptual=True
    )
    
    # Compare the hashes
    return compare_image_hashes(hash1, hash2, threshold)


def batch_calculate_hashes(images: list) -> dict:
    """
    Calculate hashes for a batch of images.
    
    Args:
        images: List of dicts with 'image_bytes' or 'image_url' keys
        
    Returns:
        Dictionary mapping image index to hash
    """
    results = {}
    for idx, image_info in enumerate(images):
        try:
            hash_value = calculate_image_hash(
                image_bytes=image_info.get('image_bytes'),
                image_url=image_info.get('image_url'),
                use_perceptual=True
            )
            results[idx] = hash_value
        except Exception as e:
            print(f"Failed to calculate hash for image {idx}: {e}")
            results[idx] = None
    
    return results