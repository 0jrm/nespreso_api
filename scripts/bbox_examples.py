#!/usr/bin/env python3
"""
BBOX Examples for NeSPReSO Grid Endpoint

This script provides examples of different BBOX configurations for querying
specific regions of the Gulf of Mexico.
"""

def print_bbox_info(name, bbox, description):
    """Print formatted BBOX information"""
    lon_min, lat_min, lon_max, lat_max = bbox
    print(f"\n{name}")
    print(f"  BBOX: {bbox}")
    print(f"  Description: {description}")
    print(f"  Format: [lon_min, lat_min, lon_max, lat_max]")
    print(f"  Range: Lon {lon_min:.2f}° to {lon_max:.2f}°, Lat {lat_min:.2f}° to {lat_max:.2f}°")

def main():
    """Display various BBOX examples for the Gulf of Mexico"""
    
    print("=== NeSPReSO Grid Endpoint - BBOX Examples ===\n")
    print("The BBOX parameter allows you to filter grid points within a specific region.")
    print("Format: [lon_min, lat_min, lon_max, lat_max]")
    print("All coordinates are in decimal degrees.\n")
    
    # Define various BBOX configurations
    bbox_examples = [
        {
            "name": "Full Gulf of Mexico",
            "bbox": None,
            "description": "Complete coverage of the Gulf of Mexico region (default, no BBOX parameter)"
        },
        {
            "name": "Western Gulf",
            "bbox": [-97.0, 20.0, -90.0, 29.0],
            "description": "Western portion including Texas and northern Mexico coasts"
        },
        {
            "name": "Eastern Gulf",
            "bbox": [-90.0, 20.0, -82.0, 29.0],
            "description": "Eastern portion including Florida and Cuba"
        },
        {
            "name": "Northern Gulf",
            "bbox": [-97.0, 25.0, -82.0, 29.0],
            "description": "Northern portion including Louisiana, Mississippi, Alabama coasts"
        },
        {
            "name": "Southern Gulf",
            "bbox": [-97.0, 18.0, -82.0, 25.0],
            "description": "Southern portion including Yucatan Peninsula and southern Mexico"
        },
        {
            "name": "Central Gulf",
            "bbox": [-94.0, 22.0, -88.0, 27.0],
            "description": "Central region of the Gulf"
        },
        {
            "name": "Texas Shelf",
            "bbox": [-96.5, 26.0, -93.5, 29.0],
            "description": "Texas continental shelf region"
        },
        {
            "name": "Florida Shelf",
            "bbox": [-87.0, 24.0, -82.0, 28.0],
            "description": "Florida continental shelf region"
        }
    ]
    
    # Display all examples
    for example in bbox_examples:
        if example["bbox"] is None:
            print(f"\n{example['name']}")
            print(f"  BBOX: None (no parameter)")
            print(f"  Description: {example['description']}")
            print(f"  Usage: Omit BBOX parameter or set to null")
        else:
            print_bbox_info(example["name"], example["bbox"], example["description"])
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES:")
    print("="*60)
    
    print("\n1. Full Grid Query (no BBOX):")
    print("   POST /v1/profile/grid")
    print("   {")
    print('     "date": "2023-01-15"')
    print("   }")
    
    print("\n2. BBOX Filtered Query:")
    print("   POST /v1/profile/grid")
    print("   {")
    print('     "date": "2023-01-15",')
    print('     "bbox": [-95.0, 20.0, -85.0, 28.0]')
    print("   }")
    
    print("\n3. Python Client Example:")
    print("   from grid_client_example import query_grid")
    print("   result = query_grid('2023-01-15', bbox=[-95.0, 20.0, -85.0, 28.0])")
    
    print("\n" + "="*60)
    print("PERFORMANCE CONSIDERATIONS:")
    print("="*60)
    print("• BBOX filtering reduces the number of grid points processed")
    print("• Smaller regions = faster processing and smaller output files")
    print("• Full grid: ~1,018 points")
    print("• Typical BBOX regions: 200-800 points")
    print("• Processing time scales with the number of points")
    
    print("\n" + "="*60)
    print("VALIDATION RULES:")
    print("="*60)
    print("• BBOX must have exactly 4 values")
    print("• Longitude: -180° to 180°")
    print("• Latitude: -90° to 90°")
    print("• lon_min < lon_max")
    print("• lat_min < lat_max")
    
    print("\n" + "="*60)
    print("OUTPUT FILES:")
    print("="*60)
    print("• Full grid: NeSPReSO_grid_2023-01-15.nc")
    print("• With BBOX: NeSPReSO_grid_2023-01-15_bbox_-95.00_20.00_-85.00_28.00.nc")

if __name__ == "__main__":
    main()
