#fuzzy set
# --- Fuzzy Set Membership Functions ---

def cold(t):
    return max(0, min(1, (20 - t) / 20))

def warm(t):
    if 15 < t < 25:
        return (t - 15) / 10
    elif 25 <= t <= 35:
        return (35 - t) / 10
    return 0

def hot(t):
    return max(0, min(1, (t - 30) / 20))

# Test
temp = 22
print("Cold =", cold(temp))
print("Warm =", warm(temp))
print("Hot  =", hot(temp))

#fuzzy relation 
import numpy as np

# Universe values
temps = [10, 20, 30, 40]
speeds = [0.2, 0.5, 0.8, 1.0]

# Memberships
def temp_hot(t):
    return max(0, min(1, (t - 25) / 15))

def fan_high(s):
    return s

# Fuzzy Relation R(x, y) = min(µ_hot(x), µ_high(y))
R = np.zeros((len(temps), len(speeds)))

for i, t in enumerate(temps):
    for j, s in enumerate(speeds):
        R[i][j] = min(temp_hot(t), fan_high(s))

print("\nFuzzy Relation Matrix:")
print(R)


#mamdani (fuzzy inf system)

import numpy as np

# ---- Membership Functions ----
def temp_hot(t):
    return max(0, min(1, (t - 25) / 15))

def fan_high(s):
    return s   # 0–1 scale

# ---- Mamdani Inference ----
def fis_output(temp):
    # Firing strength
    firing = temp_hot(temp)

    # Output universe
    S = np.linspace(0, 1, 50)

    # Implication by min rule
    output_mf = [min(firing, fan_high(s)) for s in S]

    # Defuzzification (Centroid)
    num = sum(S[i] * output_mf[i] for i in range(len(S)))
    den = sum(output_mf)

    return num / den if den != 0 else 0

# ---- Test ----
temp = 35
fan_speed = fis_output(temp)
print("\nTemperature:", temp)
print("Fan Speed Output:", round(fan_speed, 3))
