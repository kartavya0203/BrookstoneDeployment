#!/usr/bin/env python3
"""
Test script to verify area information functionality
"""

# Simulate the AREA_INFO database and get_area_information function
AREA_INFO = {
    "3bhk": {
        "super_buildup": "2650 sqft",
        "display_name": "3BHK"
    },
    "4bhk": {
        "super_buildup": "3850 sqft", 
        "display_name": "4BHK"
    },
    "3bhk_tower_duplex": {
        "super_buildup": "5300 sqft + 700 sqft carpet terrace",
        "display_name": "3BHK Tower Duplex"
    },
    "4bhk_tower_duplex": {
        "super_buildup": "7700 sqft + 1000 sqft carpet terrace",
        "display_name": "4BHK Tower Duplex" 
    },
    "3bhk_tower_simplex": {
        "super_buildup": "2650 sqft + 700 sqft carpet terrace",
        "display_name": "3BHK Tower Simplex"
    },
    "4bhk_tower_simplex": {
        "super_buildup": "3850 sqft + 1000 sqft carpet terrace", 
        "display_name": "4BHK Tower Simplex"
    }
}

def get_area_information(query):
    """Get area information from hardcoded database"""
    query_lower = query.lower()
    results = []
    
    # Check for specific unit types
    if "tower duplex" in query_lower:
        if "3bhk" in query_lower or "3 bhk" in query_lower:
            results.append(f"{AREA_INFO['3bhk_tower_duplex']['display_name']}: {AREA_INFO['3bhk_tower_duplex']['super_buildup']}")
        elif "4bhk" in query_lower or "4 bhk" in query_lower:
            results.append(f"{AREA_INFO['4bhk_tower_duplex']['display_name']}: {AREA_INFO['4bhk_tower_duplex']['super_buildup']}")
        else:
            # If tower duplex mentioned but no specific BHK, show both
            results.append(f"{AREA_INFO['3bhk_tower_duplex']['display_name']}: {AREA_INFO['3bhk_tower_duplex']['super_buildup']}")
            results.append(f"{AREA_INFO['4bhk_tower_duplex']['display_name']}: {AREA_INFO['4bhk_tower_duplex']['super_buildup']}")
    elif "tower simplex" in query_lower:
        if "3bhk" in query_lower or "3 bhk" in query_lower:
            results.append(f"{AREA_INFO['3bhk_tower_simplex']['display_name']}: {AREA_INFO['3bhk_tower_simplex']['super_buildup']}")
        elif "4bhk" in query_lower or "4 bhk" in query_lower:
            results.append(f"{AREA_INFO['4bhk_tower_simplex']['display_name']}: {AREA_INFO['4bhk_tower_simplex']['super_buildup']}")
        else:
            # If tower simplex mentioned but no specific BHK, show both
            results.append(f"{AREA_INFO['3bhk_tower_simplex']['display_name']}: {AREA_INFO['3bhk_tower_simplex']['super_buildup']}")
            results.append(f"{AREA_INFO['4bhk_tower_simplex']['display_name']}: {AREA_INFO['4bhk_tower_simplex']['super_buildup']}")
    else:
        # Regular units
        if "3bhk" in query_lower or "3 bhk" in query_lower:
            results.append(f"{AREA_INFO['3bhk']['display_name']}: {AREA_INFO['3bhk']['super_buildup']}")
        if "4bhk" in query_lower or "4 bhk" in query_lower:
            results.append(f"{AREA_INFO['4bhk']['display_name']}: {AREA_INFO['4bhk']['super_buildup']}")
    
    # If no specific BHK mentioned and no tower type, return all regular units
    if not results and not any(bhk in query_lower for bhk in ["3bhk", "4bhk", "3 bhk", "4 bhk"]) and "tower" not in query_lower:
        results = [
            f"{AREA_INFO['3bhk']['display_name']}: {AREA_INFO['3bhk']['super_buildup']}",
            f"{AREA_INFO['4bhk']['display_name']}: {AREA_INFO['4bhk']['super_buildup']}"
        ]
    
    return results

def test_area_information():
    """Test the area information functionality"""
    
    test_queries = [
        "What is the area of 3BHK?",
        "Tell me 4BHK size",
        "What's the area of all flats?",
        "3BHK tower duplex area",
        "4BHK tower simplex size",
        "Tower duplex areas",
        "Size of units",
        "Carpet area of 3BHK",
        "Super build-up area of 4BHK",
        "SBU of tower duplex"
    ]
    
    print("🧪 Testing Area Information Functionality")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        # Check if this is an area-related query
        area_keywords = ["area", "sqft", "square feet", "size", "carpet", "super build", "buildup", "built-up", "sbu"]
        is_area_query = any(keyword in query.lower() for keyword in area_keywords)
        
        if is_area_query:
            hardcoded_area_info = get_area_information(query)
            if hardcoded_area_info:
                print(f"   ✅ Found hardcoded info: {hardcoded_area_info}")
            else:
                print(f"   ⚠️ Area query but no hardcoded info found")
        else:
            print(f"   ℹ️ Not an area-related query")

if __name__ == "__main__":
    test_area_information()