import matplotlib.pyplot as plt
MAPPO = [-3.59,-3.30,-3.03,-5.97,-18.39,-27.30,-33.03,-20.75,-9.81,-3.26,0.80,0.49,1.66,0.51,3.96,-0.14,6.48,2.27,4.89,3.77,1.45,2.21,3.43,3.35,6.31,6.59,7.47,7.79,7.15,7.84,10.51,7.69,7.13,3.28,7.54,8.64,6.62,9.72,7.36,10.80,6.44,3.61,7.77,4.59,10.80,10.25,9.64,12.34,11.20,12.68]
IPPO = [-3.64,-3.49,-3.55,-3.37,-3.28,-4.74,-8.26,-5.64,-8.74,-9.43,-6.62,-10.76,-14.00,-16.93,-23.06,-20.68,-9.36,-5.77,-6.08,-2.68,-2.07,2.98,0.26,7.72,4.48,5.20,0.40,3.44,3.19,-1.71,-4.34,-6.69,-6.23,-0.64,1.62,-1.40,2.99,0.97,3.11,4.74,0.65,-2.72,-4.46,-0.03,-1.51,3.05,1.43,9.18,5.04,7.24]
CPPO = [-3.92,-4.03,-3.91,-3.74,-3.86,-3.98,-4.16,-4.28,-4.58,-9.15,-14.92,-17.26,-31.21,-22.68,-21.72,-21.07,-8.43,-4.62,-4.20,-4.21,-10.16,-1.12,-0.85,1.21,1.63,-2.12,4.86,7.83,3.43,-4.21,4.71,4.23,1.12,0.83,-4.20,-10.16,-21.10,-10.20,-2.34,1.10,0.63,-3.53,-1.20,2.05,4.73,6.44,5.13,8.42,12.63,7.42]
IPPO_improved = [-3.60,-3.88,-3.95,-3.89,-3.69,-3.49,-3.28,-2.72,-2.50,-2.50,-2.37,-2.39,-2.48,-2.61,-2.50,-2.32,-1.80,-1.21,-0.59,0.54,1.20,1.90,1.98,1.08,-0.10,0.69,1.53,2.28,2.39,2.10,1.73,1.61,1.46,6.37,1.35,3.13,4.32,7.36,9.37,8.45,4.47,6.51,8.66,11.70,13.35,13.16,12.52,11.91,15.30,14.19]

# plt.plot(MAPPO, label="MAPPO")
plt.plot(IPPO, label="IPPO")
plt.plot(IPPO_improved, label="IPPG")
# plt.plot(CPPO, label="CPPO")
print(len(CPPO))
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()

