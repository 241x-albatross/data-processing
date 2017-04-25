#This file reading data

import numpy as np
import matplotlib.pyplot as plt
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
    #drag*v + thrust*v + gravity *v = F_tot*v
    drag = (np.dot(F_tot,v_dir)   - np.dot(thrust,v_dir) )* v_dir # - np.dot(gravity,v_dir)

    lift = F_tot  - thrust - drag # - gravity

    drag_mag,lift_mag = np.linalg.norm(drag), np.linalg.norm(lift)

    ref = 0.5*rho* vel_mag**2*S

    c_drag,c_lift = drag_mag/ref, lift_mag/ref

    return drag_mag,lift_mag, c_drag, c_lift

def ComputeAOA(ADAT_u, ADAT_v, ADAT_w):
    vel = np.array([ADAT_u, ADAT_v, ADAT_w])
    #velocity direction in body coordinate
    v_dir = vel/np.linalg.norm(vel)

    alpha = 180*np.arctan(v_dir[2]/v_dir[0])/np.pi

    return alpha


def PositionVisualization(ADAT_N, ADAT_E, ADAT_D):
    plt.figure(1)
    plt.plot(ADAT_D)
    plt.xlabel('time')
    plt.ylabel('altitude')

    plt.figure(2)
    plt.plot(ADAT_N,ADAT_E)
    plt.xlabel('N')
    plt.ylabel('E')


def ChannelVisualization(time, CH0,CH1,CH2,CH3):
    plt.figure(3)
    plt.plot(time,CH0,label ="aileron")
    plt.plot(time,CH1,label ="elevator")
    plt.plot(time,CH2,label ="throttle")
    plt.plot(time,CH3,label ="rudder")
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('signal')

def NEDVelocityVisualization(time,GPS_VelN,GPS_VelE,GPS_VelD):
    plt.figure(4)
    plt.plot(time,GPS_VelN,label ="VelN")
    plt.plot(time,GPS_VelE,label ="VelE")
    plt.plot(time,GPS_VelD,label ="VelD")

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('velocity')

def BodyVelocityVisualization(time,Vx,Vy,Vz):
    plt.figure(4)
    plt.plot(time,Vx,label ="Vx")
    plt.plot(time,Vy,label ="Vy")
    plt.plot(time,Vz,label ="Vz")

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('velocity')

    plt.figure(5)
    plt.plot(time,np.sqrt(Vx**2+Vy**2+Vz**2),label ="|V|")


    plt.legend()
    plt.xlabel('time')
    plt.ylabel('velocity')


def AccVisualization(time,IMU_AccX,IMU_AccY,IMU_AccZ):
    plt.figure(6)
    plt.plot(time,IMU_AccX,label ="IMU_AccX")
    plt.plot(time,IMU_AccY,label ="IMU_AccY")
    plt.plot(time,IMU_AccZ,label ="IMU_AccZ")

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('ACC')




if __name__  == "__main__":


    """
    skip_header=5,
    """
    g = 9.8
    mass = 1.031
    rho=1.1455 #kg.m-3
    b=1.55 #m
    S=b*(0.21+0.18)/2. #m^2 rough approx!!

    ############################################
    log = np.genfromtxt('Data/log_5.csv', delimiter=",",filling_values = 0.0, skip_header=6)

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

    ADAT_u, ADAT_v, ADAT_w = log[:,458],log[:,459],log[:,460]

    ADAT_N, ADAT_E, ADAT_Dbaro,ADAT_Dgps = log[:,454],log[:,455],log[:,456],log[:,457]

    RC_C0, RC_C1, RC_C2,RC_C3 = log[:,116],log[:,117],log[:,118], log[:,119]

    BATT_V, BATT_C = log[:,178],log[:,179]

    TIME_StartTime = log[:,462]





    ##############################################################
    '''
    print(roll[0],pitch[0],yaw[0],IMU_AccX[0],IMU_AccY[0],IMU_AccZ[0],ADAT_N[0], ADAT_E[0], ADAT_Dbaro[0],ADATD_Dgps[0] )
    print(ADAT_u[0], ADAT_v[0], ADAT_w[0])
    print(RC_C0[0], RC_C1[0], RC_C2[0],RC_C3[0],  BATT_V[0], BATT_C[0],TIME_StartTime[0])
    '''

    PositionVisualization(ADAT_N, ADAT_E, ADAT_Dgps)
    ChannelVisualization(TIME_StartTime, RC_C0, RC_C1, RC_C2,RC_C3)
    BodyVelocityVisualization(TIME_StartTime,ADAT_u, ADAT_v, ADAT_w)
    AccVisualization(TIME_StartTime,IMU_AccX,IMU_AccY,IMU_AccZ)





    len = len(roll)
    drag,lift,c_drag,c_lift,alpha = np.zeros(len),np.zeros(len),np.zeros(len),np.zeros(len),np.zeros(len)
    for i in range(len):
        drag[i], lift[i], c_drag[i],c_lift[i] = ComputeDragLift(IMU_AccX[i],IMU_AccY[i],IMU_AccZ[i],
                                                                ADAT_u[i], ADAT_v[i], ADAT_w[i],
                                                                roll[i],pitch[i],yaw[i],
                                                                g,mass,rho,S)
        alpha[i] = ComputeAOA(ADAT_u[i], ADAT_v[i], ADAT_w[i])

    ############## Filter
    CH_throttle_off = RC_C2 < 1e-8

    plt.figure(100)
    plt.plot(CH_throttle_off)
    plt.legend()
    plt.title('All data')
    plt.xlabel('time')
    plt.ylabel('CH_throttle_off')


    drag_throttle_off = [drag[i] for i in range(len) if CH_throttle_off[i]]
    lift_throttle_off = [lift[i] for i in range(len) if CH_throttle_off[i]]
    c_drag_throttle_off = [c_drag[i] for i in range(len) if CH_throttle_off[i]]
    c_lift_throttle_off = [c_lift[i] for i in range(len) if CH_throttle_off[i]]
    alpha_throttle_off = [alpha[i] for i in range(len) if CH_throttle_off[i]]


    plt.figure(10)
    plt.plot(alpha_throttle_off,lift_throttle_off,'ro')
    plt.xlabel('alpha')
    plt.ylabel('lift')
    plt.title('Throttle off')
    plt.figure(11)
    plt.plot(alpha_throttle_off,drag_throttle_off,'ro')
    plt.xlabel('alpha')
    plt.ylabel('drag')
    plt.title('Throttle off')

    plt.figure(12)
    plt.plot(alpha_throttle_off,c_lift_throttle_off,'ro')
    plt.xlabel('alpha')
    plt.ylabel('lift coeff')
    plt.title('Throttle off')
    plt.figure(13)
    plt.plot(alpha_throttle_off,c_drag_throttle_off,'ro')
    plt.xlabel('alpha')
    plt.ylabel('drag coeff')
    plt.title('Throttle off')


    plt.show()
