# Figure out how to select k in k-means and check how k differs in k-means would affect the results.
import numpy as np
import matplotlib.pyplot as plt

a = np.array([21736, 19656, 18475, 17591, 16980, 16488, 16086, 15748, 15456, 15199, 14985, 14790, 14617, 14484, 14325, 14212])  # No CaseOLAP
# a = np.array([])  # CaseOLAP only
# CaseOLAP only: k=2:  19656(13855), 11885(8809),  11789(8749),  11785(8746),  11785(8746)   Very stable [0]
# CaseOLAP only: k=3:  18475(13855), 15321(11810), 15254(11769), 15250(11767), 15250(11767)  Very stable [2]
# CaseOLAP only: k=4:  17591(13855), 16030(12841), 15953(12791), 15933(12777), 15897(12753)  Relatively stable [10]
# CaseOLAP only: k=4:  17591(13855), 16035(12844), 15957(12793), 15916(12765), 15907(12759), 15904(12757), 15897(12752), 15897(12752)  Relatively stable [9]
# CaseOLAP only: k=5:  16980(13855), 15997(13208), 15920(13157), 15918(13156), 15918(13156)  Relatively stable [12] v.s. 15901(13144)
# CaseOLAP only: k=6:  16488(13855), 15592(13258), 15518(13207), 15504(13197), 15502(13196)  Relatively stable [14]
# CaseOLAP only: k=7:  16086(13855), 15340(13340), 15273(13293), 15264(13287), 15261(13285)  Relatively stable [8]
# CaseOLAP only: k=8:  15747(13855), 14973(13310), 14946(13289), 14941(13285), 14941(13285)  stable [5]
# CaseOLAP only: k=9:  15456(13855), 14797(13388), 14765(13366), 14765(13366), 14765(13366)  stable [6]
# CaseOLAP only: k=10: 15200(13855), 14632(13443), 14596(13417), 14596(13417), 14595(13416)  very stable [0]

# A second run
# k=2-10: 11785(8746), 15247(11765), 15886(12743), 15911(13151), 15480(13182), 15249(13277), 14934(13280), 14773(13372), 14595(13416)

# a = np.array([8746, 11765, 12743, 13151, 13182, 13277, 13280, 13372, 13416])  # Number of keywords
a = np.array([11785, 15247, 15886, 15911, 15480, 15249, 14934, 14773, 14595])  # Sum of distances of samples to their closest cluster center

''' Conclusion[1]: The optimal k has the larget variance after CaseOLAP. Or at least, variance doesn't drop significantly until it's > optimal k'''
''' Conclusion[2]: Number of keywords does not increase much after the optimal k'''

''' Observation[1]: k values around the optimal k will have unstable results.'''

# a = np.array([])  # CaseOLAP with filtering general terms.
''' TO-DO: Can see if filtering general terms can make our observation more significant. '''

seq = np.array(range(1, len(a)+1))
b = np.power(seq, 0)
c = np.multiply(a, b)

plt.plot(seq, c, 'ro')
plt.show()

print a
print b
print c
