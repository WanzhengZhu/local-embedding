# Figure out how to select k in k-means and check how k differs in k-means would affect the results.
import numpy as np
import matplotlib.pyplot as plt

a = np.array([21736, 19656, 18475, 17591, 16980, 16488, 16086, 15748, 15456, 15199, 14985, 14790, 14617, 14484, 14325, 14212])  # No CaseOLAP

# CaseOLAP only on the first level. threshold = 0.25
# Optimal k is 5. if k is 4 or 6, the results are not too bad.
# k=2:  19656(13855), 11885(8809),  11789(8749),  11785(8746),  11785(8746)   Very stable [0]
# k=3:  18475(13855), 15321(11810), 15254(11769), 15250(11767), 15250(11767)  Very stable [2]
# k=4:  17591(13855), 16030(12841), 15953(12791), 15933(12777), 15897(12753)  Relatively stable [10]
# k=4:  17591(13855), 16035(12844), 15957(12793), 15916(12765), 15907(12759), 15904(12757), 15897(12752), 15897(12752)  Relatively stable [9]
# k=5:  16980(13855), 15997(13208), 15920(13157), 15918(13156), 15918(13156)  Relatively stable [12] v.s. 15901(13144)
# k=6:  16488(13855), 15592(13258), 15518(13207), 15504(13197), 15502(13196)  Relatively stable [14]
# k=7:  16086(13855), 15340(13340), 15273(13293), 15264(13287), 15261(13285)  Relatively stable [8]
# k=8:  15747(13855), 14973(13310), 14946(13289), 14941(13285), 14941(13285)  stable [5]
# k=9:  15456(13855), 14797(13388), 14765(13366), 14765(13366), 14765(13366)  stable [6]
# k=10: 15200(13855), 14632(13443), 14596(13417), 14596(13417), 14595(13416)  very stable [0]

# A second run; threshold = 0.25
# k=2-10: 11785(8746), 15247(11765), 15886(12743), 15911(13151), 15480(13182), 15249(13277), 14934(13280), 14773(13372), 14595(13416)
# a = np.array([8746, 11765, 12743, 13151, 13182, 13277, 13280, 13372, 13416])  # Number of keywords
# a = np.array([11785, 15247, 15886, 15911, 15480, 15249, 14934, 14773, 14595])  # Sum of distances of samples to their closest cluster center

# A third run with threshold = 0.5
# k=2-10: 6159(4693), 11220(8891), 13542(11076), 14311(11978), 14155(12201), 14113(12398), 13875(12460), 13830(12639), 13726(12732)

# A third run with threshold = 1  (doesn't work)
# k=2-10: 0(0), 3744(3119), 6918(5955), 8571(7483), 9764(8701), 10432(9387),


# In the cluster of 'computer_vision'
# k = 2:  1353(1344)
# k = 3:  1500(1559)
# k = 4:  1497(1605) / 1501(1608)
# k = 5:  1487(1637)
# k = 6:  1454(1636)
# k = 7:  1429(1636)
# k = 8:  1405(1636)
# k = 9:  1382(1637)
# k = 10: 1361(1637)

# In the cluster of 'machine_learning'
# k = 2:  3676(3286)
# k = 3:  3886(3703)
# k = 4:  3819(3746)
# k = 5:  3754(3778)
# k = 6:  3678(3784)
# k = 7:  3605(3785)
# k = 8:  3549(3788)
# k = 9:  3498(3788)
# k = 10: 3451(3788)

# In the cluster of 'information_retrieval'
# k = 2:  2764(2475)
# k = 3:  3183(3005)
# k = 4:  3238(3170)
# k = 5:  3168(3180)
# k = 6:  3118(3199)
# k = 7:  3072(3200)
# k = 8:  3040(3222)
# k = 9:  3002(3226)
# k = 10: 2959(3228)


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
