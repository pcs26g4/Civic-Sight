import math

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in meters.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return float('inf')
        
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371000 # Radius of earth in meters
    return c * r

def group_by_location(items, distance_threshold=20):
    """
    Group items by GPS coordinates.
    items: List of dicts, each containing 'latitude' and 'longitude'.
    distance_threshold: distance in meters.
    """
    groups = []
    for item in items:
        lat = item.get('latitude')
        lon = item.get('longitude')
        
        found_group = False
        
        # Only try to group by distance if coordinates exist
        if lat is not None and lon is not None:
            for group in groups:
                rep_lat = group[0].get('latitude')
                rep_lon = group[0].get('longitude')
                
                if rep_lat is not None and rep_lon is not None:
                    dist = calculate_distance(lat, lon, rep_lat, rep_lon)
                    if dist <= distance_threshold:
                        group.append(item)
                        found_group = True
                        break
        
        if not found_group:
            groups.append([item])
    
    return groups

def get_address_details(lat, lon):
    """
    Get area and district from coordinates using Nominatim reverse geocoding.
    """
    if lat is None or lon is None:
        return {"area": "-", "district": "-", "full_address": ""}

    import requests
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1"
        headers = {
            'User-Agent': 'MDMS-Civic-Issue-Tracker/1.0'
        }
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})
            
            # Area can be suburb, neighbourhood, city_district, or town
            area = address.get('suburb') or address.get('neighbourhood') or address.get('city_district') or address.get('town') or address.get('village') or '-'
            
            # District is usually 'county' or 'state_district' in Nominatim for India
            district = address.get('state_district') or address.get('county') or address.get('district') or '-'
            
            return {
                "area": area,
                "district": district,
                "full_address": data.get('display_name', '')
            }
    except Exception as e:
        print(f"Geocoding error: {e}")
    
    return {"area": "-", "district": "-", "full_address": ""}
