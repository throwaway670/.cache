def crop_recommendation(soil_type, season):
    # Knowledge Base
    rules = {
        ("loamy", "winter"): "Wheat or Mustard",
        ("clay", "monsoon"): "Rice",
        ("sandy", "summer"): "Melons or Maize",
        ("black", "winter"): "Cotton",
    }
    
    key = (soil_type.lower(), season.lower())
    return rules.get(key, "Generic vegetables (Consult local agronomist)")

# Simulation
print("--- Crop Agent ---")
print(crop_recommendation("Clay", "Monsoon"))
print(crop_recommendation("Sandy", "Summer"))