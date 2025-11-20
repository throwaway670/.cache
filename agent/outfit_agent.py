import random

SIMULATION_DAYS = 10
COLD_TEMP_MAX = 10
MILD_TEMP_MAX = 20
WINDY_SPEED_MIN = 15
STRONG_WIND_MIN = 25

class WeatherAgent:
    def __init__(self):
        print("WeatherAgent initialized.")

    def _simulate_weather(self):
        weather = {
            'temperature_c': round(random.uniform(2, 35), 1),
            'is_raining': random.choice([True, False, False, False]),
            'wind_speed_kmh': round(random.uniform(0, 35), 1)
        }
        return weather

    def recommend_outfit(self, weather_data):
        temp = weather_data['temperature_c']
        is_raining = weather_data['is_raining']
        wind = weather_data['wind_speed_kmh']
        
        outfit = set()
        
        if temp < COLD_TEMP_MAX:
            outfit.add("Heavy Coat")
            outfit.add("Warm Sweater")
            outfit.add("Pants")
            outfit.add("Scarf and Hat")
            
        elif temp <= MILD_TEMP_MAX:
            outfit.add("Light Jacket")
            outfit.add("Long-sleeve Shirt")
            outfit.add("Pants")
        else:
            outfit.add("T-shirt")
            outfit.add("Shorts or Light Pants")
            outfit.add("Sun Hat")
            
        if is_raining:
            outfit.add("Raincoat or Umbrella")
            outfit.add("Waterproof Shoes")

        if wind >= STRONG_WIND_MIN:
            outfit.add("Windbreaker Jacket")
            outfit.add("Secure Hat/Hood")
        elif wind >= WINDY_SPEED_MIN:
            outfit.add("Light Windbreaker")

        return sorted(list(outfit))

    def run_simulation(self):
        for day in range(1, SIMULATION_DAYS + 1):
            weather = self._simulate_weather()
            outfit = self.recommend_outfit(weather)
            
            temp = f"{weather['temperature_c']}°C"
            rain_status = "Yes" if weather['is_raining'] else "No"
            wind = f"{weather['wind_speed_kmh']} km/h"

            print(f"\n--- Day {day} ---")
            print(f"Weather Report:")
            print(f"  > Temperature: {temp}")
            print(f"  > Rain: {rain_status}")
            print(f"  > Wind Speed: {wind}")
            
            print("\nRecommended Outfit:")
            if not outfit:
                print("  - Casual wear (Default)")
            else:
                for item in outfit:
                    print(f"  - {item}")

if __name__ == "__main__":
    agent = WeatherAgent()
    agent.run_simulation()
    print("\n--- Simulation Complete ---")