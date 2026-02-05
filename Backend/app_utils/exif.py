from io import BytesIO
import re

# Lazy import PIL to avoid startup errors
def _get_pil():
    try:
        from PIL import Image, ExifTags
        return Image, ExifTags
    except ImportError:
        return None, None

# Lazy import requests
def _get_requests():
    try:
        import requests
        return requests
    except ImportError:
        return None


# ---------------------------
# HELPERS
# ---------------------------
def _ratio_to_float(value):
    try:
        return value[0] / value[1]
    except Exception:
        return float(value)


def _convert_to_degrees(value):
    """
    Converts EXIF GPS coordinates to decimal degrees
    Handles ((deg,1),(min,1),(sec,100)) format
    """
    try:
        d = _ratio_to_float(value[0])
        m = _ratio_to_float(value[1])
        s = _ratio_to_float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return None


def _parse_gps_position_string(value: str):
    """
    Fallback parser for GPSPosition string
    Example: 43 deg 28' 2.81" N, 11 deg 53' 6.46" E
    """
    try:
        matches = re.findall(r"(\d+)\D+(\d+)\D+([\d.]+)", value)
        if len(matches) < 2:
            return None

        lat_d, lat_m, lat_s = map(float, matches[0])
        lon_d, lon_m, lon_s = map(float, matches[1])

        lat = lat_d + lat_m / 60 + lat_s / 3600
        lon = lon_d + lon_m / 60 + lon_s / 3600

        if "S" in value:
            lat = -lat
        if "W" in value:
            lon = -lon

        return {"latitude": lat, "longitude": lon}
    except Exception:
        return None


# ---------------------------
# EXIF GPS EXTRACTION
# ---------------------------
def _parse_semicolon_coords(value):
    """
    Parses '16; 18; 9.41...' style strings into decimal degrees
    """
    try:
        parts = str(value).split(";")
        if len(parts) >= 3:
            d = float(parts[0].strip())
            m = float(parts[1].strip())
            s = float(parts[2].strip())
            return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        pass
    return None


# ---------------------------
# EXIF GPS EXTRACTION
# ---------------------------
def extract_gps_from_image_bytes(image_bytes: bytes):
    try:
        Image, ExifTags = _get_pil()
        if Image is None:
            return None  # PIL not installed
        
        img = Image.open(BytesIO(image_bytes))
        
        # Try multiple methods to get EXIF
        exif = None
        
        # Method 1: _getexif() (Standard for JPEG)
        if hasattr(img, '_getexif'):
            exif = img._getexif()
            
        # Method 2: getexif() (Newer Pillow, supports some PNGs)
        if not exif and hasattr(img, 'getexif'):
            exif = img.getexif()

        if not exif:
            return None

        # Convert to readable dict
        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        
        # Handle GPSInfo tag (standard EXIF structure)
        gps_info = tag_map.get("GPSInfo")
        
        # Some libraries return GPSInfo as a bare dict key even if not named "GPSInfo"
        # We search specifically for the GPS tag ID (34853)
        if not gps_info:
            gps_info = exif.get(34853)

        if gps_info:
            # Map GPS keys to names
            gps_tag_map = {
                ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()
            }

            lat = lon = alt = None

            # Latitude
            if "GPSLatitude" in gps_tag_map:
                raw_lat = gps_tag_map["GPSLatitude"]
                # Try standard rational
                lat = _convert_to_degrees(raw_lat)
                # Try semicolon fallback
                if lat is None:
                    lat = _parse_semicolon_coords(raw_lat)
                
                # Apply Ref
                if lat is not None and "GPSLatitudeRef" in gps_tag_map:
                    ref = gps_tag_map["GPSLatitudeRef"]
                    if ref in ["S", "s", b"S"]:
                        lat = -lat

            # Longitude
            if "GPSLongitude" in gps_tag_map:
                raw_lon = gps_tag_map["GPSLongitude"]
                 # Try standard rational
                lon = _convert_to_degrees(raw_lon)
                # Try semicolon fallback
                if lon is None:
                    lon = _parse_semicolon_coords(raw_lon)
                
                # Apply Ref
                if lon is not None and "GPSLongitudeRef" in gps_tag_map:
                    ref = gps_tag_map["GPSLongitudeRef"]
                    if ref in ["W", "w", b"W"]:
                        lon = -lon

            # Altitude
            if "GPSAltitude" in gps_tag_map:
                alt = _ratio_to_float(gps_tag_map["GPSAltitude"])
                if gps_tag_map.get("GPSAltitudeRef", 0) in [1, b"\x01"]:
                    alt = -alt

            if lat is not None and lon is not None:
                return {
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt
                }

        # ---------- FALLBACK: GPSPosition STRING ----------
        # Sometimes specialized devices write a combined string string
        gps_position = tag_map.get("GPSPosition")
        if gps_position:
            return _parse_gps_position_string(gps_position)

        return None

    except Exception as e:
        print(f"Error extracting GPS: {e}")
        return None


# ---------------------------
# REVERSE GEOCODING
# ---------------------------
def reverse_geocode(lat: float, lon: float):
    try:
        requests = _get_requests()
        if requests is None:
            return None  # requests not installed
        
        url = "https://nominatim.openstreetmap.org/reverse"
        headers = {
            "User-Agent": "MDMS/1.0 (contact@example.com)"
        }
        params = {
            "format": "jsonv2",
            "lat": lat,
            "lon": lon
        }

        res = requests.get(url, params=params, headers=headers, timeout=10)
        if res.status_code != 200:
            return None

        data = res.json()
        addr = data.get("address", {})

        return {
            "address": data.get("display_name"),
            "city": addr.get("city") or addr.get("town") or addr.get("village"),
            "state": addr.get("state"),
            "pincode": addr.get("postcode")
        }

    except Exception:
        return None
