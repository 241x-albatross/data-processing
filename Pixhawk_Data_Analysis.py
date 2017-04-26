#This file reading data

import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
def LoadCompressedData():

    log = np.load('log_5_compressed.npy')

    return log.T #becasue we need its columns


def LoadRawData(file_name):

    ############################################
    log = np.genfromtxt(file_name, delimiter=",",filling_values = 0.0, skip_header=6)

    #Filter out the data before start off and after landing, bases on velocity bar
    ADAT_u, ADAT_v, ADAT_w = log[:,458],log[:,459],log[:,460]

    V = np.sqrt(ADAT_u**2 + ADAT_v**2 +  ADAT_w**2)

    plt.figure(0)
    plt.plot(V,label ="|V|")
    plt.legend()
    plt.title('All data')
    plt.xlabel('time')
    plt.ylabel('velocity')

    vel_bar = 3.0

    V_ind = np.where(V > vel_bar)[0]

    idx_begin, idx_end = min(V_ind), max(V_ind)
    print(len(V))
    print(idx_begin, idx_end)

    log = log[idx_begin:idx_end+1,:]

    #python start from 0
    roll,pitch,yaw = log[:,4],log[:,5],log[:,6]

    IMU_AccX,IMU_AccY,IMU_AccZ = log[:,18],log[:,19],log[:,20]

    RC_C0, RC_C1, RC_C2,RC_C3 = log[:,116],log[:,117],log[:,118], log[:,119]

    BATT_V, BATT_C = log[:,178],log[:,179]

    ADAT_N, ADAT_E, ADAT_Dbaro,ADAT_Dgps = log[:,454],log[:,455],log[:,456],log[:,457]

    ADAT_u, ADAT_v, ADAT_w = log[:,458],log[:,459],log[:,460]

    TIME_StartTime = log[:,462]

    return roll,pitch,yaw,IMU_AccX,IMU_AccY,IMU_AccZ,RC_C0, RC_C1, RC_C2,RC_C3,BATT_V, BATT_C,\
           ADAT_N, ADAT_E, ADAT_Dbaro,ADAT_Dgps,ADAT_u, ADAT_v, ADAT_w,TIME_StartTime

def RotatioMatrix(roll,pitch,yaw):
    '''
    compute rotation matrix, R, project from NED to body framework
    :param row:
    :param pitch:
    :param yaw:
    :return:
    '''
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0.0],[np.sin(yaw),np.cos(yaw),0.0],[0,0,1]])
    Ry = np.array([[np.cos(pitch),0.0,np.sin(pitch)],[0,1.0,0],[-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0.0, 0.0],[0, np.cos(roll), - np.sin(roll)],[0, np.sin(roll),np.cos(roll)]])
    return np.dot(Rz, np.dot(Ry,Rx))
    #return np.dot(Rx, np.dot(Ry,Rz))

def ComputeDragLift(IMU_AccX,IMU_AccY,IMU_AccZ, ADAT_u, ADAT_v, ADAT_w,roll, pitch, yaw, g, mass, rho, S):
    R = RotatioMatrix(roll, pitch, yaw)

    #gravity = np.dot(R,np.array([0,0,1]))*g*mass

    thrust = np.array([0.0,0.0,0.0])

    F_tot = np.array([IMU_AccX,IMU_AccY,IMU_AccZ])*mass# F_tot - Gravity is the force???


    vel = np.array([ADAT_u, ADAT_v, ADAT_w])

    vel_mag = np.linalg.norm(vel)
    if(vel_mag == 0):
        vel_mag = 1.0

    v_dir = vel/vel_mag
    l_dir = np.cross(v_dir, [0,1,0])
    l_dir /= np.linalg.norm(l_dir)

    #drag + thrust + lift  = F_tot
    drag = (np.dot(F_tot,v_dir)   - np.dot(thrust,v_dir) )* v_dir  #- np.dot(gravity,v_dir)

    lift_mag = np.dot(F_tot  - thrust - drag, l_dir)  # - gravity

    drag_mag = np.linalg.norm(drag)

    ref = 0.5*rho* vel_mag**2*S

    c_drag,c_lift = drag_mag/ref, lift_mag/ref

    return drag_mag,lift_mag, c_drag, c_lift

def ComputeAOA(ADAT_u, ADAT_v, ADAT_w):
    vel = np.array([ADAT_u, ADAT_v, ADAT_w])
    #velocity direction in body coordinate
    v_dir = vel/np.linalg.norm(vel)

    alpha = 180*np.arctan(v_dir[2]/v_dir[0])/np.pi

    return alpha

def Angle(time, roll, pitch, yaw, filter = None):
    if(filter is not None):
        roll = list(compress(roll, filter))
        pitch = list(compress(pitch, filter))
        yaw = list(compress(yaw, filter))
        time = list(compress(time, filter))

    plt.figure(-1)
    plt.plot(time,roll,'o',label = "roll",markersize=2)
    plt.plot(time, pitch,'o',label = "pitch",markersize=2)
    plt.plot(time, yaw,'o',label = "yaw",markersize=2)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('angle')

def PositionVisualization(time, ADAT_N, ADAT_E, ADAT_D,filter = None):
    if(filter is not None):
        ADAT_N = list(compress(ADAT_N, filter))
        ADAT_E = list(compress(ADAT_E, filter))
        ADAT_D = list(compress(ADAT_D, filter))
        time = list(compress(time, filter))


    plt.figure(1)
    plt.plot(time, ADAT_D,'o',markersize=2)
    plt.xlabel('time')
    plt.ylabel('altitude')

    plt.figure(2)
    plt.plot(ADAT_N,ADAT_E,'o',markersize=2)
    plt.xlabel('N')
    plt.ylabel('E')


def ChannelVisualization(time, CH0,CH1,CH2,CH3,filter = None):
    if(filter is not None):
        CH0 = list(compress(CH0, filter))
        CH1 = list(compress(CH1, filter))
        CH2 = list(compress(CH2, filter))
        CH3 = list(compress(CH3, filter))
        time = list(compress(time, filter))

    plt.figure(3)
    plt.plot(time,CH0,'-o',label ="aileron",markersize=2)
    plt.plot(time,CH1,'-o',label ="elevator",markersize=2)
    plt.plot(time,CH2,'-o',label ="throttle",markersize=2)
    plt.plot(time,CH3,'-o',label ="rudder",markersize=2)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('signal')

def NEDVelocityVisualization(time,GPS_VelN,GPS_VelE,GPS_VelD,filter = None):
    plt.figure(4)
    plt.plot(time,GPS_VelN,label ="VelN",markersize=2)
    plt.plot(time,GPS_VelE,label ="VelE",markersize=2)
    plt.plot(time,GPS_VelD,label ="VelD",markersize=2)

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('velocity')

def BodyVelocityVisualization(time,Vx,Vy,Vz,filter = None):
    if(filter is not None):
        Vx = np.array(list(compress(Vx, filter)))
        Vy = np.array(list(compress(Vy, filter)))
        Vz = np.array(list(compress(Vz, filter)))
        time = np.array(list(compress(time, filter)))


    plt.figure(4)
    plt.plot(time,Vx,'o',label ="Vx",markersize=2)
    plt.plot(time,Vy,'o',label ="Vy",markersize=2)
    plt.plot(time,Vz,'o',label ="Vz",markersize=2)

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('velocity')

    plt.figure(5)
    plt.plot(time, np.sqrt(Vx**2+Vy**2+Vz**2),'-o',label ="|V|",markersize=2)


    plt.legend()
    plt.xlabel('time')
    plt.ylabel('velocity')


def AccVisualization(time,IMU_AccX,IMU_AccY,IMU_AccZ,filter = None):
    if(filter is not None):
        IMU_AccX = list(compress(IMU_AccX, filter))
        IMU_AccY = list(compress(IMU_AccY, filter))
        IMU_AccZ = list(compress(IMU_AccZ, filter))
        time = list(compress(time, filter))

    plt.figure(6)
    plt.plot(time,IMU_AccX,'-o',label ="IMU_AccX",markersize=2)
    plt.plot(time,IMU_AccY,'-o',label ="IMU_AccY",markersize=2)
    plt.plot(time,IMU_AccZ,'-o',label ="IMU_AccZ",markersize=2)

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('ACC')

'''
def get_max_DL(lift, drag, throttle, V, e=0.85, b=1.55,S=1.55*(0.21+0.18)/2., rho=1.1455 ):
        Cd0=np.zeros((lift.shape[0],1))
        for i in range(lift.shape[0]):
            if Filter_off[i]:
                vel=V[i]
                Cd=2*drag[i]/(rho*vel**2.)
                Cl=2*lift[i]/(rho*vel**2.)
                Cd0[i]=Cd-Cl**2./(np.pi*S*e)

        Cd0_avg=np.sum(Cd0)/np.count_nonzero(Cd0)
        DL_max=.5*np.sqrt(np.pi*b**2./S/Cd0_avg)
        return DL_max
'''
def PowerConsumption(Batt_VFilt,Batt_CFilt,Vx,Vy,Vz,Filter_off):


    power = Batt_VFilt*Batt_CFilt
    V = np.sqrt(Vx**2 + Vy**2 + Vz**2)

    plt.figure(7)
    plt.plot(V, power, 'bo',markersize=2)
    plt.title('power consumption')
    plt.xlabel('velocity')
    plt.ylabel('power')

    plt.figure(8)
    power = power*Filter_off
    plt.plot(V, power,'bo',markersize=2)
    plt.title('throttle-off level flight power')
    plt.xlabel('velocity')
    plt.ylabel('power')


if __name__  == "__main__":

    g = 9.8
    mass = 1.031
    rho=1.1455 #kg.m-3
    b=1.55 #m
    S=b*(0.21+0.18)/2. #m^2 rough approx!!

    roll,pitch,yaw,IMU_AccX,IMU_AccY,IMU_AccZ,RC_C0, RC_C1, RC_C2,RC_C3,BATT_V, BATT_C,\
           ADAT_N, ADAT_E, ADAT_Dbaro,ADAT_Dgps,ADAT_u, ADAT_v, ADAT_w,TIME_StartTime = LoadCompressedData()

    IMU_AccZ = -IMU_AccZ

    ##############################################################
    '''
    print(roll[0],pitch[0],yaw[0],IMU_AccX[0],IMU_AccY[0],IMU_AccZ[0],ADAT_N[0], ADAT_E[0], ADAT_Dbaro[0],ADATD_Dgps[0] )
    print(ADAT_u[0], ADAT_v[0], ADAT_w[0])
    print(RC_C0[0], RC_C1[0], RC_C2[0],RC_C3[0],  BATT_V[0], BATT_C[0],TIME_StartTime[0])
    '''

    ############## Filter
    data_len = len(roll)

    Filter_off = [(RC_C2[i] < 1e-5)  for i in range(data_len)]


    Angle(TIME_StartTime, roll, pitch, yaw,Filter_off)
    PositionVisualization(TIME_StartTime,ADAT_N, ADAT_E, ADAT_Dgps,Filter_off)
    ChannelVisualization(TIME_StartTime, RC_C0, RC_C1, RC_C2,RC_C3,Filter_off)
    BodyVelocityVisualization(TIME_StartTime,ADAT_u, ADAT_v, ADAT_w,Filter_off)
    AccVisualization(TIME_StartTime,IMU_AccX,IMU_AccY,IMU_AccZ,Filter_off)






    drag,lift,c_drag,c_lift,alpha = np.zeros(data_len),np.zeros(data_len),np.zeros(data_len),np.zeros(data_len),np.zeros(data_len)
    for i in range(data_len):
        drag[i], lift[i], c_drag[i],c_lift[i] = ComputeDragLift(IMU_AccX[i],IMU_AccY[i],IMU_AccZ[i],
                                                                ADAT_u[i], ADAT_v[i], ADAT_w[i],
                                                                roll[i],pitch[i],yaw[i],
                                                                g,mass,rho,S)
        alpha[i] = ComputeAOA(ADAT_u[i], ADAT_v[i], ADAT_w[i])




    #Filter_off = [(RC_C2[i] < 1e-5) and (np.fabs(roll[i]) < 5*np.pi/180) for i in range(data_len)]



    #Filter_acc=[np.linalg.norm(np.array([IMU_AccX,IMU_AccY,IMU_AccZ]))< 10 for i in range(data_len)]



    plt.figure(100)
    plt.plot(Filter_off,markersize=2)
    plt.title('All data')
    plt.xlabel('time')
    plt.ylabel('Filter_off')

    PowerConsumption(BATT_V, BATT_C, ADAT_u, ADAT_v, ADAT_w,Filter_off)


    #Filter_off value is true for n1 n1+1 .. n1+k1, n2 n2+1... n2+k2 .....
    #we need to find ni and ki

    Filter_segments = [i for i in range(data_len) if Filter_off[i] and ((i==0 or not Filter_off[i-1]) or ((i==data_len-1) or not Filter_off[i+1]))]



    print(len(Filter_segments)//2, Filter_segments)
    seg_id = 6
    if seg_id >0:
        filter_start_id = Filter_segments[2*seg_id-2]
        filter_end_id = Filter_segments[2 * seg_id - 1]+1
    else:
        filter_start_id = 0
        filter_end_id = data_len


    drag_filtered = [drag[i] for i in np.arange(filter_start_id,filter_end_id) if Filter_off[i]]
    lift_filtered = [lift[i] for i in np.arange(filter_start_id,filter_end_id) if Filter_off[i]]
    c_drag_filtered = [c_drag[i] for i in np.arange(filter_start_id,filter_end_id) if Filter_off[i]]
    c_lift_filtered = [c_lift[i] for i in np.arange(filter_start_id,filter_end_id) if Filter_off[i]]
    alpha_filtered = [alpha[i] for i in np.arange(filter_start_id,filter_end_id) if Filter_off[i]]


    plt.figure(10)
    plt.plot(alpha_filtered,lift_filtered,'ro',markersize=2)
    plt.xlabel('alpha')
    plt.ylabel('lift')
    plt.title('Throttle off')
    plt.figure(11)
    plt.plot(alpha_filtered,drag_filtered,'ro',markersize=2)
    plt.xlabel('alpha')
    plt.ylabel('drag')
    plt.title('Throttle off')

    plt.figure(12)
    plt.plot(alpha_filtered,c_lift_filtered,'ro',markersize=2)
    plt.xlabel('alpha')
    plt.ylabel('lift coeff')
    plt.title('Throttle off')
    plt.figure(13)
    plt.plot(alpha_filtered,c_drag_filtered,'ro',markersize=2)
    plt.xlabel('alpha')
    plt.ylabel('drag coeff')
    plt.title('Throttle off')

    plt.figure(14)
    plt.plot(alpha_filtered,np.array(c_lift_filtered)/np.array(c_drag_filtered),'ro',markersize=2)
    plt.xlabel('alpha')
    plt.ylabel('CL/CD')
    plt.title('Throttle off')


    plt.show()

