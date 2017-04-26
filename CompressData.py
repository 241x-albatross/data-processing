import numpy as np
############################################
log = np.genfromtxt('../Data/log_5.csv', delimiter=",",filling_values = 0.0, skip_header=6)

#Filter out the data before start off and after landing, bases on velocity bar
ADAT_u, ADAT_v, ADAT_w = log[:,458],log[:,459],log[:,460]

V = np.sqrt(ADAT_u**2 + ADAT_v**2 +  ADAT_w**2)


vel_bar = 3.0

V_ind = np.where(V > vel_bar)[0]

idx_begin, idx_end = min(V_ind), max(V_ind)
print(len(V))
print(idx_begin, idx_end)

log = log[idx_begin:idx_end+1,:]

#python start from 0
'''
roll,pitch,yaw = log[:,4],log[:,5],log[:,6]

IMU_AccX,IMU_AccY,IMU_AccZ = log[:,18],log[:,19],log[:,20]

RC_C0, RC_C1, RC_C2,RC_C3 = log[:,116],log[:,117],log[:,118], log[:,119]

BATT_V, BATT_C = log[:,178],log[:,179]

ADAT_N, ADAT_E, ADAT_Dbaro,ADAT_Dgps = log[:,454],log[:,455],log[:,456],log[:,457]

ADAT_u, ADAT_v, ADAT_w = log[:,458],log[:,459],log[:,460]

TIME_StartTime = log[:,462]
'''
compressed_log = log[:,[4,5,6,18,19,20,116,117,118,119,178,179,454,455,456,457,458,459,460,462]]
np.save('log_5_compressed',compressed_log)