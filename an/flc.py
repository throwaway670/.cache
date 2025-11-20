import numpy as np

# Triangular membership function
def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif x == b:
        return 1
    elif x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

# 1. Compute error
T_set = 30
T_measured = 26
error = T_set - T_measured   # 4
print("Error =", error)

# 2. Membership values
mu_N = triangular(error, -10, -5, 0)
mu_Z = triangular(error, -5, 0, 5)
mu_P = triangular(error, 0, 5, 10)

print("Membership Values:")
print("N =", mu_N)
print("Z =", mu_Z)
print("P =", mu_P)

# 3. Apply rules → firing strengths
# Rule: N->High, Z->Medium, P->Low
w_high = mu_N
w_med  = mu_Z
w_low  = mu_P

print("\nFiring Strengths:")
print("High  =", w_high)
print("Medium=", w_med)
print("Low   =", w_low)

# 4. Defuzzification (Weighted centroid)
Low = 20
Med = 50
High = 80


# ✔ 1. Membership Values for error = 4

# N(-10, -5, 0)
# 4 is outside → μN = 0

# Z(-5, 0, 5)
# (4 is between 0 and 5 → falling side)
# μZ = (5 − 4) / (5 − 0) = 0.2

# P(0, 5, 10)
# (4 is between 0 and 5 → rising side)
# μP = (4 − 0) / (5 − 0) = 0.8
# ✔ 2. Firing Strengths (By rules)

#     If N → High → firing strength = μN = 0

#     If Z → Medium → firing strength = μZ = 0.2

#     If P → Low → firing strength = μP = 0.8

# ✔ 3. Defuzzification (Weighted Centroid)
# Output=(0.8)(20)+(0.2)(50)+(0)(80)0.8+0.2+0
# Output=0.8+0.2+0(0.8)(20)+(0.2)(50)+(0)(80)​
# =16+101=26
# =116+10​=26
# ⭐ FINAL ANSWERS
# Membership Values

#     μN = 0

#     μZ = 0.2

#     μP = 0.8

# Firing Strengths

#     Low = 0.8

#     Medium = 0.2

#     High = 0

# Defuzzified Output

# 👉 26
# numerator = (w_low * Low) + (w_med * Med) + (w_high * High)
# denominator = (w_low + w_med + w_high)

# if denominator == 0:
#     output = 0✔ 1. Membership Values for error = 4

# N(-10, -5, 0)
# 4 is outside → μN = 0

# Z(-5, 0, 5)
# (4 is between 0 and 5 → falling side)
# μZ = (5 − 4) / (5 − 0) = 0.2

# P(0, 5, 10)
# (4 is between 0 and 5 → rising side)
# μP = (4 − 0) / (5 − 0) = 0.8
# ✔ 2. Firing Strengths (By rules)

#     If N → High → firing strength = μN = 0

#     If Z → Medium → firing strength = μZ = 0.2

#     If P → Low → firing strength = μP = 0.8

# ✔ 3. Defuzzification (Weighted Centroid)
# Output=(0.8)(20)+(0.2)(50)+(0)(80)0.8+0.2+0
# Output=0.8+0.2+0(0.8)(20)+(0.2)(50)+(0)(80)​
# =16+101=26
# =116+10​=26
# ⭐ FINAL ANSWERS
# Membership Values

#     μN = 0

#     μZ = 0.2

#     μP = 0.8

# Firing Strengths

#     Low = 0.8

#     Medium = 0.2

#     High = 0

# Defuzzified Output

# 👉 26
# else:
#     output = numerator / denominator

# print("\nDefuzzified Output =", output)
